import ROOT
import numpy as np
'''                                                                                                                                                                                                                                          Make a ROOT tree from the condensed numpy arrays
'''
    
arrays = np.load('data/PixelChangesCondensedData_sorted.npz')
sec = arrays['seconds']
eventNb = arrays['eventNumber']
canvasX = arrays['pixelXpos'].astype('int32')
canvasY = arrays['pixelYpos'].astype('int32')
col = arrays['colorIndex'].astype('int32')
user = arrays['userIndex']
modEvent = arrays['moderatorEvent'].astype('int32')
 
dataframe = ROOT.RDF.MakeNumpyDataFrame({'time': sec, 'pixelXpos':canvasX, 'pixelYpos':canvasY, 'eventNb': eventNb, 'color': col, 'user': user, 'moderatorEvent':modEvent })
 
# You can now use the RDataFrame as usually, e.g. add a column ...
#df = df.Define('z', 'x + y')
 
dataframe.Snapshot('tree', 'data/PixelChanges.root')
