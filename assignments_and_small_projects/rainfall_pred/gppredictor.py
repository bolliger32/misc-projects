import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin

from pyproj import Proj

from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY,
                       AltitudeMode, Camera)


class GPPredictor(BaseEstimator,RegressorMixin):
    
    def __init__(self,h=50000,sigma_n=0,proj=None,shuffle=None):
        
        self.h = h  # gaussian bandwidth parameter
        self.sigma_n = sigma_n  # measurement noise
        self._proj = proj

    # when fitting, assume no measurement noise unless specified
    def fit(self,X,Y):
        self.adj, self.X = self.utm_and_center(X)
        self.adj_y, self.Y = self.center_data(Y)
        self.K = self.covariance(self.X,self.X,self.h)
    
    def predict(self,X_test):
        Xt = self.to_utm(X_test,self.adj)
        K_xt_x ,K_x_xt, K_xt_xt = self.get_K_matrices(Xt,self.X,self.h)
        
        # calculate mean prediction using (1) (in assignment description)
        result = np.dot(np.dot(K_xt_x,np.linalg.inv(self.K + 
                                                    self.sigma_n**2 * 
                                                    np.eye(self.K.shape[0]))),
                        self.Y) - self.adj_y
        
        return result
    
    def simulate(self,X_grid,gamma=.001,random_seed=None,n_cell=None):
        if random_seed != None:
            np.random.seed(random_seed)
            
        if type(X_grid) == list:
            X_grid = self.make_grid(bounding_box = X_grid,ncell = n_cell)

        Xt = self.to_utm(X_grid,adj=self.adj)
        K_xt_x ,K_x_xt, K_xt_xt = self.get_K_matrices(Xt,self.X,self.h)
        
        cov_f = K_xt_xt - np.dot(K_xt_x,
                                 np.dot(np.linalg.inv(self.K +
                                                        self.sigma_n**2 *
                                                        np.eye(self.K.shape[0])),
                                        K_x_xt))
        
        L = np.linalg.cholesky(cov_f + gamma*np.eye(cov_f.shape[0]))
        u = np.random.multivariate_normal(np.zeros(L.shape[0]),np.eye(L.shape[0]))
        
        f_sim = self.predict(X_grid) + np.expand_dims(np.dot(L,u),axis=1)
        return f_sim
        
    def visualize(self,X_grid,n_cell=None,fname='rainfall.png'):
        if type(X_grid) == list:
            bounding_box = X_grid
            X_grid = self.make_grid(bounding_box = bounding_box, ncell = n_cell)
        else:
            bounding_box = [X_grid[:,0].min(),X_grid[:,0].max(),X_grid[:,1].min(),X_grid[:,1].max()]
            
        f_sim = self.simulate(X_grid,random_seed=0)
        lat = X_grid[:,0].reshape(50,50)
        lon = X_grid[:,1].reshape(50,50)
        rain = f_sim.reshape(50,50)
        
        pixels = 1024 * 10

        fig, ax = self.gearth_fig(bounding_box=bounding_box,
                            pixels=pixels)
        ax.pcolormesh(lon, lat, rain)
        ax.set_axis_off()
        fig.savefig(fname, transparent=False, format='png')
        self.make_kml(bounding_box, [fname])
        return fig, ax
        
    # ripped from https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/
    def gearth_fig(self,bounding_box, pixels=1024):
        """Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image."""
        llcrnrlon = bounding_box[2]
        llcrnrlat = bounding_box[0]
        urcrnrlon = bounding_box[3]
        urcrnrlat = bounding_box[1]
        aspect = np.cos(np.mean([llcrnrlat, urcrnrlat]) * np.pi/180.0)
        xsize = np.ptp([urcrnrlon, llcrnrlon]) * aspect
        ysize = np.ptp([urcrnrlat, llcrnrlat])
        aspect = ysize / xsize
    
        if aspect > 1.0:
            figsize = (10.0 / aspect, 10.0)
        else:
            figsize = (10.0, 10.0 * aspect)
    
        if False:
            plt.ioff()  # Make `True` to prevent the KML components from poping-up.
        fig = plt.figure(figsize=figsize,
                        frameon=False,
                        dpi=pixels//10)
        # KML friendly image.  If using basemap try: `fix_aspect=False`.
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(llcrnrlon, urcrnrlon)
        ax.set_ylim(llcrnrlat, urcrnrlat)
        return fig, ax
    
    # ripped from https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/    
    def make_kml(self,bounding_box, figs, colorbar=None, **kw):
        """TODO: LatLon bbox, list of figs, optional colorbar figure,
        and several simplekml kw..."""
    
        llcrnrlon = bounding_box[2]
        llcrnrlat = bounding_box[0]
        urcrnrlon = bounding_box[3]
        urcrnrlat = bounding_box[1]
        
        kml = Kml()
        altitude = kw.pop('altitude', 2e7)
        roll = kw.pop('roll', 0)
        tilt = kw.pop('tilt', 0)
        altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
        camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                        longitude=np.mean([urcrnrlon, llcrnrlon]),
                        altitude=altitude, roll=roll, tilt=tilt,
                        altitudemode=altitudemode)
    
        kml.document.camera = camera
        draworder = 0
        for fig in figs:  # NOTE: Overlays are limited to the same bbox.
            draworder += 1
            ground = kml.newgroundoverlay(name='GroundOverlay')
            ground.draworder = draworder
            ground.visibility = kw.pop('visibility', 1)
            ground.name = kw.pop('name', 'overlay')
            ground.color = kw.pop('color', '9effffff')
            ground.atomauthor = kw.pop('author', 'ocefpaf')
            ground.latlonbox.rotation = kw.pop('rotation', 0)
            ground.description = kw.pop('description', 'Matplotlib figure')
            ground.gxaltitudemode = kw.pop('gxaltitudemode',
                                        'clampToSeaFloor')
            ground.icon.href = fig
            ground.latlonbox.east = llcrnrlon
            ground.latlonbox.south = llcrnrlat
            ground.latlonbox.north = urcrnrlat
            ground.latlonbox.west = urcrnrlon
    
        if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).
            screen = kml.newscreenoverlay(name='ScreenOverlay')
            screen.icon.href = colorbar
            screen.overlayxy = OverlayXY(x=0, y=0,
                                        xunits=Units.fraction,
                                        yunits=Units.fraction)
            screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                    xunits=Units.fraction,
                                    yunits=Units.fraction)
            screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                        xunits=Units.fraction,
                                        yunits=Units.fraction)
            screen.size.x = 0
            screen.size.y = 0
            screen.size.xunits = Units.fraction
            screen.size.yunits = Units.fraction
            screen.visibility = 1
    
        kmzfile = kw.pop('kmzfile', 'overlay.kmz')
        kml.savekmz(kmzfile)
        
    def get_K_matrices(self,X_test,X_train,h):
        K_xt_x = self.covariance(X_test,X_train,h)
        K_x_xt = K_xt_x.T
        K_xt_xt = self.covariance(X_test,X_test,h)
        return K_xt_x,K_x_xt,K_xt_xt
    
    def utm_and_center(self,coords):
        return self.center_data(self.to_utm(coords))
    
    def to_utm(self,coords,adj=0):
        """X must be in [lat,lng] format"""
        if not self._proj:
            self.proj = Proj("+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84")
        
        x,y = self.proj(coords[:,1],coords[:,0])
        
        result = np.zeros(coords.shape)
        result[:,0] = x
        result[:,1] = y
        result = result + adj
        
        return result
        
    def center_data(self,arr):
        adjustments = -arr.mean(axis=0)
        return adjustments,arr+adjustments

    def covariance(self, X, Z, h):
        d = spatial.distance_matrix(X,Z)
        K = np.exp(-(d**2) / (2*h*h))
        return K
        
    def make_grid(self,bounding_box, ncell):
        xmin, xmax, ymax, ymin = bounding_box
        xgrid = np.linspace(xmin, xmax, ncell)
        ygrid = np.linspace(ymin, ymax, ncell)
        mX, mY = np.meshgrid(xgrid, ygrid)
        ngridX = mX.reshape(ncell*ncell, 1);
        ngridY = mY.reshape(ncell*ncell, 1);
        return np.concatenate((ngridX, ngridY), axis=1)
        
if __name__=='__main__':
    from sklearn.grid_search import GridSearchCV
        
    def load_data(fname, delimiter=',', skiprows=1, shuffle=False, test=False,random_seed=0):
        data = np.genfromtxt(fname, delimiter=delimiter, skiprows=skiprows)
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(data)
        if test:
            X = data
            return X
        else:
            X, Y = data[:,:-1], data[:,-1:]
            return X,Y
    
    def get_new_search_limits(grid,selected_param):
        """Returns the adjacent grid elements to the optimal parameter from a grid search"""
        ix = np.nonzero(grid == selected_param)[0][0]
        return grid[ix-1],grid[ix+1]

    ## 1. Prediction
    X,Y = load_data('trn_data.csv',shuffle=True)
    X_test = load_data('tst_locations.csv',test=True)
    predictor = GPPredictor()
    
    #grid search
    grid_length=20
    np.random.seed(0)
    
    search_grid_h = np.logspace(0,6,grid_length)
    search_grid_sigma = np.logspace(-3,1,grid_length)
    pred = GridSearchCV(GPPredictor(),param_grid={'h':search_grid_h,'sigma_n':search_grid_sigma},cv=10,n_jobs=-1)
    pred.fit(X,Y)
    best_params = pred.best_params_
    
    best_h = best_params['h']
    best_sigma = best_params['sigma_n']
    
    
    # iterate linear grid search with finer mesh until the change in optimal h is less than a threshold
    threshold_h = 100.
    threshold_sigma = .01
    diff_h = threshold_h + 1.
    diff_sigma = threshold_sigma + 1.
    iterations = 1
    
    while (diff_h > threshold_h) or (diff_sigma > threshold_sigma):
        print "iteration #:", iterations
        print "current best h:", best_h
        print "current best sigma_n:", best_sigma
        print "current score:", pred.best_score_
        print ""
        
        search_lim_h = get_new_search_limits(search_grid_h,best_h)
        search_grid_h = np.linspace(search_lim_h[0],search_lim_h[1],grid_length)
        search_lim_sigma = get_new_search_limits(search_grid_sigma,best_sigma)
        search_grid_sigma = np.linspace(search_lim_sigma[0],search_lim_sigma[1],grid_length)
        
        pred = GridSearchCV(GPPredictor(),param_grid={'h':search_grid_h,'sigma_n':search_grid_sigma},cv=10,n_jobs=-1)
        pred.fit(X,Y)
        best_params = pred.best_params_
        
        best_h_old = best_h
        best_h = best_params['h']
        best_sigma_old = best_sigma
        best_sigma = best_params['sigma_n']
        
        
        iterations += 1
        diff_h = abs(best_h - best_h_old)
        diff_sigma = abs(best_sigma - best_sigma_old)
        
        
    # predict using optimized bandwidth parameter
    y_vals = pred.predict(X_test)
    prediction = np.hstack((X_test,y_vals))
    np.savetxt('predictions.csv',prediction,header='lat,lon,mm_predicted',delimiter=',',fmt='%.3f',comments='')
    
    
    ## 2. Simulation
    bounding_box = [38.5, 39.3, -120.8, -119.8]
    estimator = pred.best_estimator_
    f_sim = estimator.simulate(bounding_box,random_seed=0,n_cell=50)


    ## 3. Visualization
    estimator.visualize(bounding_box,n_cell=50)