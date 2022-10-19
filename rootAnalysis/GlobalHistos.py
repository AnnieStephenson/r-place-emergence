import sys
sys.argv.append( '-b-' ) #fno graphical output (batch mode)
import ROOT
import numpy as np
from array import array

def DrawHeatMap(h, Zmin, Zmax, title, logz):
    
    h.SetMaximum(Zmax)
    h.SetMinimum(Zmin)
    h.SetTitle(';pixel X; pixel Y;# of pixel changes')
    
    h.GetZaxis().SetTitleOffset(1.3)

    h.GetYaxis().SetTitleOffset(1.5)
    h.GetYaxis().SetTickLength(0)
    h.GetYaxis().SetLabelSize(0)
    h.GetYaxis().CenterTitle()

    h.GetXaxis().SetTitleOffset(1.1)
    h.GetXaxis().SetLabelSize(0.033)
    h.GetXaxis().SetLabelOffset(0.02)
    h.GetXaxis().SetTickLength(0)
    h.GetXaxis().SetNdivisions(10)
    h.GetXaxis().CenterTitle()

    leftm = 0.11
    rightm = 0.15
    bottomm = 0.05
    topm = 0.1
    c1 = ROOT.TCanvas('c1','c1',(int)(2020/(1-leftm-rightm)), (int)(2020/(1-bottomm-topm))) ##### try to get the exact number of pixels
    c1.SetCanvasSize(3000,2800);
    #c1.SetWindowSize(500, 500);
    c1.SetMargin(leftm,rightm,bottomm,topm)
    if logz:
        c1.SetLogz()
    
    h.Draw('X+COLZ')

    ROOT.gPad.Update()

    newaxis = ROOT.TGaxis(ROOT.gPad.GetUxmin(),
                          ROOT.gPad.GetUymax(),
                          ROOT.gPad.GetUxmin(),
                          ROOT.gPad.GetUymin(),
                          h.GetYaxis().GetXmin(),
                          h.GetYaxis().GetXmax(),
                          10,"R-")
    newaxis.SetLabelSize(0.033)
    newaxis.SetLabelOffset(-0.01)
    newaxis.SetLabelFont(42) 
    newaxis.SetTickLength(0.)##### not working...
    newaxis.Draw()

    c1.SaveAs('figs/'+title)



arrays = np.load('data/PixelChangesCondensedData_sorted.npz')
sec = arrays['seconds']
eventNb = arrays['eventNumber']
canvasX = arrays['pixelXpos'].astype('int32')
canvasY = arrays['pixelYpos'].astype('int32')
col = arrays['colorIndex'].astype('int32')
user = arrays['userIndex']
modEvent = arrays['moderatorEvent'].astype('int32')

###### create dataframe from numpy arrays
print('make dataframe')
df_pre = ROOT.RDF.MakeNumpyDataFrame({'time': sec, 'pixelXpos':canvasX, 'pixelYp':canvasY, 'eventNb': eventNb, 'color': col, 'user': user, 'moderatorEvent':modEvent })
df = df_pre.Define('pixelYpos','1999-pixelYp') ###### revert Y axis

ROOT.gStyle.SetOptStat(0) ##### no statistics box on canvas

maxtime = 3.01e5

##### Histograms of time of pixel changes

print('make time histogram')
h_time = df.Histo1D(("h_time", "h_time", 1000, 0, maxtime), 'time')
print('make time histogram (no moderator)')
h_time_noMod = df.Filter('moderatorEvent==0').Histo1D(("h_time", "h_time", 1000, 0, maxtime),'time')
print('make time histogram (moderator only)')
h_time_modEvents = df.Filter('moderatorEvent==1').Histo1D(("h_time", "h_time", 1000, 0, maxtime),'time')

h_time.SetTitle(';time [sec];# of pixel changes')
h_time_noMod.SetTitle(';time [sec];# of pixel changes')
h_time_modEvents.SetTitle(';time [sec];# of pixel changes')

c_time = ROOT.TCanvas('c_time','c_time',2000,2000)
c_time.SetMargin(0.14,0.08,0.1,0.05) ##### left,right,bottom,top

print('Draw time histogram')
h_time.Draw()
c_time.SaveAs('figs/TimeOfPixelChanges.pdf')

print('Draw time histogram (no moderator)')
h_time_noMod.Draw()
c_time.SaveAs('figs/TimeOfPixelChanges_noModerator.pdf')

print('Draw time histogram (moderator only)')
h_time_modEvents.Draw()
c_time.SaveAs('figs/TimeOfPixelChanges_moderatorEvents.pdf')


###### Heat map 2D histos

print('make heat map 2D histogram')
heatMap = df.Histo2D(('heatmap','heatmap',2020,-10,2010,2020,-10,2010),'pixelXpos','pixelYpos')

print('draw heat map')
ROOT.gStyle.SetPalette(60)##### 105: kThermometer, 60: kBlueRedYellow, 69: kBeach (white top)
DrawHeatMap(heatMap,0.99,2001,'HeatMap.bmp',True)

DrawHeatMap(heatMap,499,20000,'HeatMap_MoreThan500PixelChanges.png',True)
ROOT.gStyle.SetPalette(69)##### 105: kThermometer, 60: kBlueRedYellow, 69: kBeach (white top)
DrawHeatMap(heatMap,-0.01,5.5,'HeatMap_LessThan5PixelChanges.png',False)
ROOT.gStyle.SetPalette(60)##### 105: kThermometer, 60: kBlueRedYellow, 69: kBeach (white top)


print('make heat map vs time 3D histogram')
heatMap3D = df.Histo3D(('heatmap','heatmap',2020,-10,2010,2020,-10,2010,30,0,maxtime),'pixelXpos','pixelYpos','time')

zbinsize = float(maxtime/heatMap3D.GetNbinsZ())
for i in range(1,heatMap3D.GetNbinsZ()+1):
    heatMap3D.GetZaxis().SetRange(i,i)
    DrawHeatMap(heatMap3D.Project3D('yx'),0.99, 1+(2000/heatMap3D.GetNbinsZ()), 'timeDep/HeatMap_time{:06d}to{:06d}.png'.format(int((i-1)*zbinsize), int(i*zbinsize)),False)
#can use ImageJ to easily make a movie out of these time-dependent images

print('make heat map for moderator events')
heatMap_mod = df.Filter('moderatorEvent==1').Histo2D(('heatmap_mod','heatmap_mod',2020,-10,2010,2020,-10,2010),'pixelXpos','pixelYpos')
DrawHeatMap(heatMap_mod, 0.01,3, 'HeatMap_moderatorEvents.png',False)

strongUser=14822
print('make heat map for user',strongUser)
heatMap_mod = df.Filter('user=='+str(strongUser)).Histo2D(('heatmap_mod','heatmap_mod',2020,-10,2010,2020,-10,2010),'pixelXpos','pixelYpos')
DrawHeatMap(heatMap_mod, 0.01, heatMap_mod.GetMaximum(), 'HeatMap_user'+str(strongUser)+'.png',False)



print('count # of pixel changes per user')
###### make array with number of counts for each user (one user at each index of the array). This is faster with np.bincount in principle, but cumbersome with the [not modEvent] condition
nmaxusers = int(1.05e7)
usercounts = [0]*nmaxusers
for i in range(0,user.size):
    if not modEvent[i]: ###### remove moderator events
        usercounts[user[i]] += 1

###### Fill each array element (=each count of pixel changes per user) in histogram
PixelsPerUser = ROOT.TH1I('pixperuser','pixperuser',900,1,901)
for i in range(0, len(usercounts)):
    PixelsPerUser.Fill(usercounts[i]) ###### adds the number of counts for this user as a histogram entry
    if usercounts[i]>700:
        print('user with more than 700 pixel changes:' ,i)

c2 = ROOT.TCanvas('c2','c2',2000,2000)
PixelsPerUser.SetTitle(';# of pixel changes;# of users')
PixelsPerUser.GetYaxis().SetRangeUser(0.5,1.3*PixelsPerUser.GetMaximum())
PixelsPerUser.SetLineWidth(1)
PixelsPerUser.Draw()

c2.SetMargin(0.1,0.05,0.1,0.04) ##### left,right,bottom,top
c2.SetLogy()
c2.SetLogx()

c2.SaveAs('figs/PixelChangesPerUser_noModerator.pdf')



print('distribution of # of changes per pixel')
###### make array with number of counts for each user (one user at each index of the array). This is faster with np.bincount in principle, but cumbersome with the [not modEvent] condition
pixelCounts = [0]*(2000*2000)
for i in range(10,heatMap.GetNbinsX()-10):
    for j in range(10,heatMap.GetNbinsY()-10):
        #if not modEvent[i]: ###### remove moderator events
        pixelCounts[(i-10)*2000+(j-10)] = heatMap.GetBinContent(i,j)

###### Fill each array element (=each count of pixel changes per user) in histogram
binning = [0.5]
for i in range(1,300):
    binning.append(i)
for i in range(30,90):
    binning.append(i*10)
for i in range(9,50):
    binning.append(i*100)
for i in range(5,75):
    binning.append(i*1000)

ChangesPerPix = ROOT.TH1D('changesperpix','changesperpix',len(binning)-1,array('d',binning))
for i in range(0, len(pixelCounts)):
    ChangesPerPix.Fill(pixelCounts[i] if pixelCounts[i]>0.5 else 0.51 ) ###### adds the number of counts for this user as a histogram entry
    if pixelCounts[i]>10000:
        print('pixel with more than 10000 changes: x,y =' ,i//2000,i%2000, ',',int(pixelCounts[i]),'pixel changes')

for i in range(1,ChangesPerPix.GetNbinsX()+1):
    if ChangesPerPix.GetBinWidth(i)>1:
        ChangesPerPix.SetBinContent(i, ChangesPerPix.GetBinContent(i) / ChangesPerPix.GetBinWidth(i))
        
c2 = ROOT.TCanvas('c2','c2',2000,2000)
ChangesPerPix.SetTitle(';# of changes;# of pixels / bin width')
ChangesPerPix.GetYaxis().SetRangeUser(0.5/1000,1.3*ChangesPerPix.GetMaximum())
ChangesPerPix.GetYaxis().SetTitleOffset(1.2)
ChangesPerPix.GetXaxis().SetTitleOffset(1.2)
ChangesPerPix.SetLineWidth(1)
ChangesPerPix.Draw()

c2.SetMargin(0.1,0.05,0.1,0.04) ##### left,right,bottom,top
c2.SetLogy()
c2.SetLogx()

c2.SaveAs('figs/ChangesPerPixel.pdf')

