import numpy as np
from scipy.optimize import least_squares
from astropy.coordinates import SkyCoord


class FitSphericalEllipse():
    
    def __init__(self, edge_coords, skycoord_init, semi_major_init, semi_minor_init, theta_init):
        
        """
        Initialize the instance.
        
        Parameters
        ----------
        edge_coords : astropy.coordinates.SkyCoord
            The galactic coordinates of the edge pixels.
        skycoord0 : astropy.coordinates.SkyCoord
            The initiall guess for the center of the localization region.
        a0 : float
            The initial guess for the semi-major axes.
        b0 : float
            The initial guess for the semi-minor axes.
        theta0 : float
            The initial guess for the rotation angle of the localization region.
        """
        
        
        # convert the edge sky coordinates to 3D cartesian coordinates
        self.edge_coords = edge_coords
        self.edge_cartesian = FitSphericalEllipse.skycoord2cartesian(self.edge_coords)
        
            
        # convert degrees to radians. The fitting always use radians internally
        self.lon_init_rad = np.deg2rad(skycoord_init.l.deg)
        self.lat_init_rad = np.deg2rad(skycoord_init.b.deg)
        self.polar_init_rad = np.pi/2 - self.lat_init_rad  # the fitting is performs in the spherical system, so we need polar angle
        self.a_init = np.tan(np.deg2rad(semi_major_init))  # convert the semi-major axis (degrees) to a parameters (distance in the tangent space)
        self.b_init = np.tan(np.deg2rad(semi_minor_init))
        self.theta_init_rad = np.deg2rad(theta_init)
        
    def fit_ellipse(self, lon_bound = [0, 360], lat_bound = [-90, 90], a_bound = [0, 1], b_bound = [0, 1], theta_bound = [0, 180]):
        
        # The user input the latitude bound, it converts to polar ang
        lon_bound_rad = np.deg2rad(lon_bound)
        lat_bound_rad = np.deg2rad(lat_bound)
        theta_bound_rad = np.deg2rad(theta_bound)
        
        # pay attention to the range of polar angles
        param_bounds = ([lon_bound_rad[0], np.pi/2 - lat_bound_rad[1], a_bound[0], b_bound[0], theta_bound_rad[0]],
                        [lon_bound_rad[1], np.pi/2 - lat_bound_rad[0], a_bound[1], b_bound[1], theta_bound_rad[1]])
        
        result = least_squares(FitSphericalEllipse.cost_function, 
                               [self.lon_init_rad, self.polar_init_rad, self.a_init, self.b_init, self.theta_init_rad], 
                               args=(self.edge_cartesian,), 
                               method='trf',  # Trust Region Reflective: more stable for nonlinear fitting
                               jac='3-point',  # Improves numerical derivatives
                               xtol=1e-12,  # Tighter tolerance for more accuracy
                               ftol=1e-12,
                               max_nfev=10000, bounds=param_bounds)  # Allow more iterations
                               # bounds for longitude: [0, 360] ([0, 2*pi])
                               # bounds for latitude: [-90, 90] ([-pi/2, pi/2])
                               # bounds for a: [0, 10]
                               # bounds for b: [0, 10]
                               # theta for b: [0, 180] [0, pi]
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
            
        else:
        
            self.lon_fit_rad, self.polar_fit_rad, self.a_fit, self.b_fit, self.theta_fit_rad = result.x
            
            self.lat_fit_rad = np.pi/2 - self.polar_fit_rad
            
            return self.lon_fit_rad, self.lat_fit_rad, self.a_fit, self.b_fit, self.theta_fit_rad
        
    
    @property
    def ellipse_param(self):
        
        center_coord = SkyCoord(l = np.rad2deg(self.lon_fit_rad), b = np.rad2deg(self.lat_fit_rad), unit = "deg", frame = "galactic")
        
        semi_major = np.rad2deg(np.arctan(self.a_fit))
        
        semi_minor = np.rad2deg(np.arctan(self.b_fit))
        
        theta_deg = np.rad2deg(self.theta_fit_rad)
        
        print(f"Fitted Center (l, b): ({center_coord.l.deg:.2f}, {center_coord.b.deg:.2f})")
        print(f"Fitted Axes (a, b): ({semi_major:.2f} deg, {semi_minor:.2f} deg)")
        print(f"Fitted Position Angle θ: {theta_deg:.2f} deg")
        
        return center_coord, semi_major, semi_minor, theta_deg
    
    def get_ellipse_edge(self, npoints = 100):
        
        return FitSphericalEllipse.generate_spherical_ellipse(self.lon_fit_rad, 
                                                              self.lat_fit_rad, 
                                                              self.a_fit, 
                                                              self.b_fit, 
                                                              self.theta_fit_rad, 
                                                              npoints = npoints)
        
        
    @staticmethod
    def generate_spherical_ellipse(lon, lat, a, b, theta, npoints = 100):
        
        """
        Generate the data points along the spherical ellipse formed by the intersection of a ellipse cone and a unit sphere.
        
        Parameters
        ----------
        lon : float
            The longitude of the center of the spherical ellipse. The unit is radians.
        lat : float
            The latitude of the center of the spherical ellipse. The unit is radians.
        a : float
            The semi-major axes of the ellipse cone. Note that this is the Euclidean space distance in the tagent plane.
        b : float
            The semi-minor axes of the ellise cone. Note that this is the Euclidean space distance in the tagent plane.
        theta : float
            The rotation angle of the spherical ellipse.
        npoints : int
            The number of data points on the spherical ellipse.
            
        Returns
        -------
        astropy.coordinates.SkyCoord
            The sky coordinate of the data points
        
        """
        
        # the center of the spherical ellipse
        # I decided to use radians internally, so conversion is not need
        # I kept this variable assignment to aviod changing the code below and remind myself that lon, lat and theta are in radians.
        lon_rad = lon
        polar_rad = np.pi/2 - lat
        theta_rad = theta
        
        # get the rotation matrix rotate the ellipse from the standard position (north pole) to the localization region
        R_rotation = FitSphericalEllipse.construct_R(lon_rad, polar_rad, theta_rad)
        
        # The phi angles for the spherical ellipse data points
        lon_edge_rad = np.linspace(0, 2 * np.pi, npoints)
        
        # The theta angles solved by the phi angles
        tan_polar_edge = np.reciprocal(np.sqrt((np.cos(lon_edge_rad)**2)/(a**2) + (np.sin(lon_edge_rad)**2)/(b**2)))
        polar_edge_rad = np.arctan(tan_polar_edge)
        
        # get Cartesian coordinates for the standard spherical ellipse
        x_local = np.sin(polar_edge_rad)*np.cos(lon_edge_rad)
        y_local = np.sin(polar_edge_rad)*np.sin(lon_edge_rad)
        z_local = np.cos(polar_edge_rad)
        vec_local = np.vstack([x_local, y_local, z_local]).T
        
        
        # rotate the 3D cartesian coordinates of the standard ellipse
        vec_global = (R_rotation @ vec_local.T).T
        
        # calculate the radii, they are all supposed to be 1
        r = np.sqrt(vec_global[:,0]**2 + vec_global[:,1]**2 + vec_global[:,2]**2)
        
        # the longitude for the spherical ellipse data points
        rotated_theta_edge_rad = np.arccos(vec_global[:,2] / r)
        rotated_lat_edge_rad = np.pi/2 - rotated_theta_edge_rad
        lat_edge_deg = np.rad2deg(rotated_lat_edge_rad)
        
        # the latitude for the spherical ellipse data points
        rotated_lon_edge_rad = np.arctan2(vec_global[:,1], vec_global[:,0])
        rotated_lon_edge_deg = np.rad2deg(rotated_lon_edge_rad)
        
        # the sky coordintes in degrees
        edge_skycoords = SkyCoord(l = rotated_lon_edge_deg, b = lat_edge_deg, unit = "deg", frame = "galactic")
        
        return edge_skycoords
    
        
    @staticmethod
    def skycoord2cartesian(skycoords):
        
        lon_rad = np.radians(skycoords.l.deg)
        polar_rad = np.radians(90) - np.radians(skycoords.b.deg)
        x = np.sin(polar_rad) * np.cos(lon_rad)
        y = np.sin(polar_rad) * np.sin(lon_rad)
        z = np.cos(polar_rad)
        return np.vstack([x, y, z]).T
        
    @staticmethod
    def cost_function(params, edge_cartesian):
        
        lon_init_rad, polar_init_rad, a_init, b_init, theta_init_rad = params
        Q = FitSphericalEllipse.construct_Q(lon_init_rad, polar_init_rad, a_init, b_init, theta_init_rad)
        
        # convert the edge sky coordinates to cartesian
        # edge_cartesian is a n*3 numpy array
        
        residuals = []
        for x in edge_cartesian:
            x = x[:, None] # make it a vertical vector
            residual = x.T @ Q @ x  # Quadratic form x^T Q x, it returns a (1,1) matrix
            residual = residual[0][0]  # extract the value
            residuals.append(residual**2)  # Square to penalize large deviations
            
        return np.array(residuals)
        
    @staticmethod
    def construct_R(lon, polar, theta):
        
        """
        Construct the rotation matrix R that rotate the standard spherical ellipse to where the localization region is.
        We can rotate the coordinates of the standard spherical ellipse by:
        X_global = R @ X_local
        or
        X_local = R.T @ X_global
        
        Parameters
        ----------
        lon : float
            The lontitude of the rotated center. The unit is radians.
        polar : float
            The polar angle (co-latitude) of the rotated center. The unit is radians.
        a : float
            The semi-major axes of the ellipse cone. Note that this is the Euclidean space distance in the tagent plane.
        b : float
            The semi-minor axes of the ellise cone. Note that this is the Euclidean space distance in the tagent plane.
        theta : float
            The rotation angle of the localization region. The unit is radians.
            
        Returns
        -------
        numpy.ndarray
            The rotation matrix of the spherical ellipse from the standard position to the localization region.
        """
        
        # I decided to use radians internally, so conversion is not need
        # I kept this variable assignment to aviod changing the rotation matrix below and remind myself that lon, lat and theta are in radians.
        lon_rad = lon
        polar_rad = polar
        theta_rad = theta
        
        #1. Rotate by position angle theta around z-axis
        R_theta = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                            [np.sin(theta_rad), np.cos(theta_rad),  0],
                            [0,                 0,                  1]])
    
        # 2. Rotate by (90° - b0) around y-axis to tilt to latitude b0
        R_b = np.array([[np.cos(polar_rad),  0, np.sin(polar_rad)],
                        [0,                  1,                 0],
                        [-np.sin(polar_rad), 0, np.cos(polar_rad)]])
    
        # 3. Rotate by l0 around z-axis to set longitude
        R_l = np.array([[np.cos(lon_rad), -np.sin(lon_rad), 0],
                        [np.sin(lon_rad), np.cos(lon_rad),  0],
                        [0,              0,                 1]])
    
        # Combined rotation: R = R_l * R_b * R_theta
        R = R_l @ R_b @ R_theta
    
        return R
        
        
    @staticmethod
    def construct_Q(lon, polar, a, b, theta):
        
        """
        Contruct the matrix of the quadratic form, Q, from the ellipse parameters.
        In our case the quadratic form is the surface of the elliptic cone: X_global^T @ Q @ X_global = 0.
        X_global is the 3D Cartisian coordinates with the coordinate system sitting at the center of the sphere.
        
        Parameters
        ----------
        lon : float
            The lontitude of the rotated center. The unit is radians.
        polar : float
            The polar angle (co-latitude) of the rotated center. The unit is radians.
        a : float
            The semi-major axes of the ellipse cone. Note that this is the Euclidean space distance in the tagent plane.
        b : float
            The semi-minor axes of the ellise cone. Note that this is the Euclidean space distance in the tagent plane.
        theta : float
            The rotation of the spherical ellipse. The unit is radians.
    
        
        Returns
        -------
        numpy.ndarray
            The ellipse rotation matrix.
        """
        
        # convert angles to radians
        R = FitSphericalEllipse.construct_R(lon, polar, theta)
        
        # The matrix of the quadratic form at the standard position (the center axis is aligned with the z axis.)
        S = np.diag([1.0 / (a**2), 1.0 / (b**2), -1.0])
        
        # The matrix of the quadratic form
        Q = R @ S @ R.T
        
        return Q
        
        
        
        
    