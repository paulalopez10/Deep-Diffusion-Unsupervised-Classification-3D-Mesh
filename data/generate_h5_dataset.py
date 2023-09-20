# -*- coding: utf-8 -*-
import numpy as np
import h5py
import pointsampler
import off
from glob import glob
from argparse import ArgumentParser

def readList( list_filename ) :
    file = open( list_filename )
    list = file.readlines()
    file.close()
    for i in range( len( list ) ) :
        list[ i ] = list[ i ].rstrip("\n")
    return list

def readMap( map_filename ) :
    map = {}
    map_file = open( map_filename, 'r' )
    for line in map_file :
        itemlist = line[:-1].split(' ')
        map[ itemlist[ 0 ] ] = itemlist[ 1 ]
    map_file.close()
    return map

def main(hparams):

    in_dirname = hparams.in_dirname # directory that contains 3D shapes 
    n_point = int(hparams.n_point) # number of 3D points sampled per 3D shape
    labels = hparams.labels # label file that contains correspondences between a name of 3D shape and a name of category
    classes = hparams.classes # label file that contains a list of category names
    out_filepath = hparams.out_filepath # output file path

    model2category = readMap( labels )
    categories = readList( classes )

    category2id = {}
    for i in range( len( categories ) ) :
        category2id[ categories[ i ] ] = i

    Points = []
    Labels = []
    Names = []

    ps = pointsampler.PointSampler( n_point )

    n_shape = len( model2category )
    count = 0

    for i in model2category.items() :

        if( count % 10 == 0 ) :
            print( str(count) + " / " + str( n_shape ) )
        count += 1

        modelname = i[ 0 ]
        ctgname = i[ 1 ]
        ctgid = category2id[ ctgname ]

        file = glob(in_dirname+ "/" + modelname + '*')
        if  file == []:
            print( "Error: " + in_dirname + "/" + modelname  + " not found" )

            quit()
        else:
            print(file)
            print(file[0])
            in_filepath = file[0]


        mesh = off.Mesh( in_filepath )
        oript = ps.sample( mesh )

        pos = oript[ :, 0:3 ]
        nor = oript[ :, 3:6 ]

        # Normalizes position
        mean = np.mean( pos, axis=0 )
        pos = pos - mean

        # Normalizes scale of a 3D model so that it is enclosed by a sphere whose radius is 0.5
        radius = 0.5
        norms = np.linalg.norm( pos, axis=1 )
        max_norm = np.max( norms )
        pos = pos * ( radius / max_norm )

        oript = np.hstack( [ pos, nor ] )

        Points.append( oript )
        Labels.append( ctgid )
        Names.append( modelname )

    Points = np.array( Points, dtype=np.float32 )
    Labels = np.array( Labels, dtype=np.int32 )
    Names = np.array( Names, dtype="S" )

    # Shuffle randomly
    idx = np.arange( len( model2category ) )
    np.random.shuffle( idx )
    Points = Points[ idx ]
    Labels = Labels[ idx ]
    Names = Names[ idx ]

    # write to a file
    f = h5py.File( out_filepath, 'w' )
    f.create_dataset( 'data', data=Points, dtype=np.float32 )
    f.create_dataset( 'label', data=Labels, dtype=np.int32 )
    f.create_dataset( 'name', data=Names )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_dirname', default='./data/meshes')
    parser.add_argument('--n_point', default=1024)
    parser.add_argument('--labels', default='./data/labels.txt')
    parser.add_argument('--classes', default='./data/unique_labels.txt')
    parser.add_argument('--out_filepath', default='./data/training_set-h5' )
    args = parser.parse_args()
    main(args)