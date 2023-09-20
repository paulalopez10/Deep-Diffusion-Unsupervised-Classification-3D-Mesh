# -*- coding: utf-8 -*-
import numpy as np
import sobol
from off import *

class PointSampler :
    def __init__( self, n_point ) :
        self.n_point = n_point;
        self.seed = 1;

    def sample( self, mesh ) :
        pos = np.zeros( ( self.n_point, 3 ), dtype=np.float32 );
        ori = np.zeros( ( self.n_point, 3 ), dtype=np.float32 );

        sa = mesh.area_total / float( self.n_point ); 

        error = 0.0;
        count = 0;

        for i in range( mesh.n_face ) :
            tmp = mesh.area_face[ i ]; 

           
            npt = 0;
            tmp /= sa;
            tmp2 = tmp;
            while( tmp >= 1 ) :
                tmp -= 1;
            error += tmp2 - npt;  
            if( error >= 1 ) :    
                npt += 1;
                error -= 1;

            for j in range( npt ) :
                vec, self.seed = sobol.i4_sobol( 2, self.seed );
                r1 = np.sqrt( vec[ 0 ] );
                r2 = vec[ 1 ];
                for k in range( 3 ) : # 
                    pos[ count ][ k ] = \
                    ( 1.0-r1 ) * mesh.vert[ mesh.face[ i ][ 0 ] ][ k ] + \
                    r1 * ( 1.0 - r2 ) * mesh.vert[ mesh.face[ i ][ 1 ] ][ k ] + \
                    r1 * ( r2 * mesh.vert[ mesh.face[ i ][ 2 ] ][ k ] );
                ori[ count ] = mesh.norm_face[ i ]; #
                count += 1;

        if( count != self.n_point ) :  # 
            vec, self.seed = sobol.i4_sobol( 2, self.seed );
            r1 = np.sqrt( vec[ 0 ] );
            r2 = vec[ 1 ];
            for k in range( 3 ) : # Osada
                pos[ self.n_point - 1 ][ k ] = \
                ( 1.0-r1 ) * mesh.vert[ mesh.face[ mesh.n_face - 1 ][ 0 ] ][ k ] + \
                r1 * ( 1.0 - r2 ) * mesh.vert[ mesh.face[ mesh.n_face - 1 ][ 1 ] ][ k ] + \
                r1 * ( r2 * mesh.vert[ mesh.face[ mesh.n_face - 1 ][ 2 ] ][ k ] );
            ori[ self.n_point - 1 ] = mesh.norm_face[ mesh.n_face - 1 ]; # 
            count += 1;

        return np.hstack( [ pos, ori ] );


if( __name__ == "__main__" ) :
    mesh = Mesh( "T0.off" );
    pointsampler = PointSampler( 2048 );
    oript = pointsampler.sample( mesh );
    np.savetxt( "out.xyz", oript );
    quit();
