import sys
from numpy import *
import scipy, scipy.special

from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Util(object):
    _errstr = "Mode is unknown or incompatible with input array shape."
    def __init__(self):
        pass
    @staticmethod
    def featureNormalize( data ):
        mu             = mean( data, axis=0 )
        data_norm     = data - mu
        sigma         = std( data_norm, axis=0, ddof=1 )
        data_norm     = data_norm / sigma
        return data_norm, mu, sigma

    @staticmethod
    def sigmoid( z ):
        # return array(handythread.parallel_map( lambda z: 1.0 / (1.0 + exp(-z)), z ))
        return scipy.special.expit(z)
        

    @staticmethod
    def sigmoidGradient( z ):
        sig = Util.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def recodeLabel( y, k ):
        m = shape(y)[0]
        out = zeros( ( k, m ) )
        for i in range(0, m):
            out[y[i]-1, i] = 1
        return out

    @staticmethod
    def mod( length, divisor ):
        dividend = array([x for x in range(1, length+1)])
        divisor  = array([divisor for x in range(1, length+1)])
        return mod( dividend, divisor ).reshape(1, length )

    @staticmethod
    def fmincg( f, x0, fprime, args, maxiter=100 ):
        nargs = (x0,) + args

        realmin = finfo(double).tiny
        RHO = 0.01     # a bunch of constants for line searches
        SIG = 0.5      # rho and sig are the constants in the wolfe-powell conditions
        INT = 0.1      # don't reevaluate within 0.1 of the limit of the current bracket
        EXT = 3.0      # extrapolate maximum 3 times the current bracket
        MAX = 20       # max 20 function evaluations per line search
        RATIO = 100    # maximum allowed slope ratio5
        length = maxiter

        red = 1
        i     = 0                                 # zero the run length counter
        ls_failed = False                           # no previous line search has failed
        fX     = array([])
        f1     = f(*nargs)                        # get function value and gradient
        df1 = fprime(*nargs)
        i     = i + (length<0)                    # count epochs?!
        s     = -df1                              # search direction is steepest
        d1     = -s.T.dot(s)                       # this is the slope
        z1     = red/(1-d1)                        # initial step is red/(|s|+1)


        while ( i < abs( length )):
            i         = i + (length>0)
            X0         = copy( x0 )
            f0         = copy( f1 )
            df0     = copy( df1 )

            x0         = x0 + (z1 * s).reshape( shape( x0 )[0], 1 )
            nargs     = (x0,) + args
            f2         = f( *nargs )
            df2     = fprime( *nargs)

            i         = i + (length<0)
            d2         = df2.T.dot(s)

            
            f3 = copy(f1)    # initialize point 3 equal to point 1
            d3 = copy(d1) 
            z3 = copy(-z1)

            M = MAX if length > 0 else min( MAX, -length-i )
            success = False
            limit = -1

            while True:
                while ((f2 > f1 + z1 * RHO * d1) or (d2 > -SIG * d1)) and ( M > 0 ):
                    limit = z1
                    if f2 > f1:
                        z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)
                    else:
                        A = 6*(f2-f3)/z3+3*(d2+d3)                      # make cubic extrapolation
                        B = 3*(f3-f2)-z3*(d3+2*d2)
                        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A
                    
                    if isnan( z2 ) or isinf( z2 ):
                        z2 = z3 / 2

                    z2 = max(min( z2, INT*z3 ), (1-INT)* z3)
                    z1 = z1 + z2
                    x0 = x0 + (z2 * s).reshape( shape( x0 )[0], 1 )
                    nargs     = (x0,) + args
                    f2         = f( *nargs )
                    df2     = fprime( *nargs)
                    
                    M = M - 1
                    i = i + (length<0)
                    
                    d2 = df2.T.dot( s )     # numerically unstable here, but the value still stays as very small decimal number
                    z3 = z3 - z2

                if (f2 > f1 + z1 * RHO * d1 ) or (d2 > -SIG * d1 ):
                    break
                elif d2 > SIG * d1:
                    success = True
                    break
                elif M == 0:
                    break

                A = 6*(f2-f3)/z3+3*(d2+d3)
                B = 3*(f3-f2)-z3*(d3+2*d2)
                z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3))
                  
                if not isreal(z2) or isnan(z2) or isinf(z2) or z2 < 0:
                    if limit < -0.5:
                          z2 = z1 * (EXT - 1)
                    else:
                          z2 = (limit - z1) / 2
                elif (limit > -0.5) and (z2+z1 > limit):
                    z2 = (limit-z1)/2
                elif (limit < -0.5) and (z2+z1 > z1 * EXT ):
                    z2 = z1 * (EXT - 1.0)
                elif z2 < -z3 * INT:
                    z2 = -z3 * INT
                elif (limit > -0.5) and (z2 < (limit-z1) * (1.0-INT)):
                    z2 = (limit-z1) *(1.0-INT)
                
                f3 = copy( f2 )
                d3 = copy( d2 )
                z3 = copy( -z2 )
                z1 = z1 + z2
                x0 = x0 + (z2 * s).reshape( shape( x0 )[0], 1 )
                nargs     = (x0,) + args
                f2         = f( *nargs )
                df2     = fprime( *nargs)
                
                M = M - 1
                i = i + (length<0)
                d2 = df2.T.dot( s )
                
            if success is True:
                f1 = copy( f2 )
                
                tmp = []
                tmp[len(tmp):] = fX.tolist()
                tmp[len(tmp):] = [f1.tolist()]
                fX = array(tmp)
                
                s = (df2.T.dot(df2) - df1.T.dot(df2)) / (df1.T.dot(df1)) * s - df2
                
                tmp = copy( df1 )
                df1 = copy( df2 )
                df2 = copy( df1 )
                d2 = df1.T.dot( s )
                
                if d2 > 0:
                    s = -df1
                    d2 = -s.T.dot( s )

                z1 = z1 * min(RATIO, d1 / (d2-realmin))
                d1 = copy(d2)
                ls_failed = False
            else:
                x0 = copy( X0 )
                f1 = copy( f0 )
                df1 = copy( df0 )

                if ls_failed is True or i > abs(length):
                    break

                tmp = copy( df1 )
                df1 = copy( df2 )
                df2 = copy( tmp )

                s = -df1
                d1 = -s.T.dot( s )
                z1 = 1 / (1 - d1)
                ls_failed = True
                
        return x0, fX
    @staticmethod
    def bytescale(data, cmin=None, cmax=None, high=255, low=0):
        """
        Byte scales an array (image).
        Byte scaling means converting the input image to uint8 dtype and scaling
        the range to ``(low, high)`` (default 0-255).
        If the input image already has dtype uint8, no scaling is done.
        This function is only available if Python Imaging Library (PIL) is installed.
        Parameters
        ----------
        data : ndarray
            PIL image data array.
        cmin : scalar, optional
            Bias scaling of small values. Default is ``data.min()``.
        cmax : scalar, optional
            Bias scaling of large values. Default is ``data.max()``.
        high : scalar, optional
            Scale max value to `high`.  Default is 255.
        low : scalar, optional
            Scale min value to `low`.  Default is 0.
        Returns
        -------
        img_array : uint8 ndarray
            The byte-scaled array.
        Examples
        --------
        >>> from scipy.misc import bytescale
        >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
        ...                 [ 73.88003259,  80.91433048,   4.88878881],
        ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
        >>> bytescale(img)
        array([[255,   0, 236],
               [205, 225,   4],
               [140,  90,  70]], dtype=uint8)
        >>> bytescale(img, high=200, low=100)
        array([[200, 100, 192],
               [180, 188, 102],
               [155, 135, 128]], dtype=uint8)
        >>> bytescale(img, cmin=0, cmax=255)
        array([[91,  3, 84],
               [74, 81,  5],
               [52, 34, 28]], dtype=uint8)
        """
        if data.dtype == np.uint8:
            return data

        if high > 255:
            raise ValueError("`high` should be less than or equal to 255.")
        if low < 0:
            raise ValueError("`low` should be greater than or equal to 0.")
        if high < low:
            raise ValueError("`high` should be greater than or equal to `low`.")

        if cmin is None:
            cmin = data.min()
        if cmax is None:
            cmax = data.max()

        cscale = cmax - cmin
        if cscale < 0:
            raise ValueError("`cmax` should be larger than `cmin`.")
        elif cscale == 0:
            cscale = 1

        scale = float(high - low) / cscale
        bytedata = (data - cmin) * scale + low
        return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

    @staticmethod
    def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
                mode=None, channel_axis=None):
        """Takes a numpy array and returns a PIL image.
        This function is only available if Python Imaging Library (PIL) is installed.
        The mode of the PIL image depends on the array shape and the `pal` and
        `mode` keywords.
        For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
        (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
        is given as 'F' or 'I' in which case a float and/or integer array is made.
        .. warning::
            This function uses `bytescale` under the hood to rescale images to use
            the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
            It will also cast data for 2-D images to ``uint32`` for ``mode=None``
            (which is the default).
        Notes
        -----
        For 3-D arrays, the `channel_axis` argument tells which dimension of the
        array holds the channel data.
        For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
        by default or 'YCbCr' if selected.
        The numpy array must be either 2 dimensional or 3 dimensional.
        """
        data = np.asarray(arr)
        if np.iscomplexobj(data):
            raise ValueError("Cannot convert a complex-valued array.")
        shape = list(data.shape)
        valid = len(shape) == 2 or ((len(shape) == 3) and
                                    ((3 in shape) or (4 in shape)))
        if not valid:
            raise ValueError("'arr' does not have a suitable array shape for "
                             "any mode.")
        if len(shape) == 2:
            shape = (shape[1], shape[0])  # columns show up first
            if mode == 'F':
                data32 = data.astype(np.float32)
                image = Image.frombytes(mode, shape, data32.tostring())
                return image
            if mode in [None, 'L', 'P']:
                bytedata = Util.bytescale(data, high=high, low=low,
                                     cmin=cmin, cmax=cmax)
                image = Image.frombytes('L', shape, bytedata.tostring())
                if pal is not None:
                    image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                    # Becomes a mode='P' automagically.
                elif mode == 'P':  # default gray-scale
                    pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                           np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                    image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                return image
            if mode == '1':  # high input gives threshold for 1
                bytedata = (data > high)
                image = Image.frombytes('1', shape, bytedata.tostring())
                return image
            if cmin is None:
                cmin = np.amin(np.ravel(data))
            if cmax is None:
                cmax = np.amax(np.ravel(data))
            data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
            if mode == 'I':
                data32 = data.astype(np.uint32)
                image = Image.frombytes(mode, shape, data32.tostring())
            else:
                raise ValueError(_errstr)
            return image

        # if here then 3-d array with a 3 or a 4 in the shape length.
        # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
        if channel_axis is None:
            if (3 in shape):
                ca = np.flatnonzero(np.asarray(shape) == 3)[0]
            else:
                ca = np.flatnonzero(np.asarray(shape) == 4)
                if len(ca):
                    ca = ca[0]
                else:
                    raise ValueError("Could not find channel dimension.")
        else:
            ca = channel_axis

        numch = shape[ca]
        if numch not in [3, 4]:
            raise ValueError("Channel axis dimension is not valid.")

        bytedata = Util.bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
        if ca == 2:
            strdata = bytedata.tostring()
            shape = (shape[1], shape[0])
        elif ca == 1:
            strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
            shape = (shape[2], shape[0])
        elif ca == 0:
            strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
            shape = (shape[2], shape[1])
        if mode is None:
            if numch == 3:
                mode = 'RGB'
            else:
                mode = 'RGBA'

        if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
            raise ValueError(_errstr)

        if mode in ['RGB', 'YCbCr']:
            if numch != 3:
                raise ValueError("Invalid array shape for mode.")
        if mode in ['RGBA', 'CMYK']:
            if numch != 4:
                raise ValueError("Invalid array shape for mode.")

        # Here we know data and mode is correct
        image = Image.frombytes(mode, shape, strdata)
        return image


import unittest
class TestUtil(unittest.TestCase):
    def setUp( self ):
        pass

    def tearDown( self ) :
        pass

    def test_sigmoid( self ):
        self.assertEqual( Util.sigmoid( 0 ), 0.5 )
        
def main():
    print(Util.sigmoid( array([0, 0, 1]) ))
    print(Util.sigmoidGradient( array([0, 0, 1]) ))

    # y = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print Util.recodeLabel( y, 10 )

    pass

    # print mat1
    # print Util.ravelMat( mat1, 3, 3 )

if __name__ == '__main__':
    # unittest.main()
    main()
