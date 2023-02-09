import ROOT
from GlobalHistos import DrawHeatMap
import pickle
import os
import numpy as np
import sys
sys.path.insert(1, os.path.join(os.getcwd(),'rplacem'))
from . import Variables as var

t_lims = [0, TIME_WHITEONLY]
file_path = os.path.join(os.getcwd(), 'data', 'stability_time{:06d}to{:06d}.pickle'.format(int(t_lims[0]), int(t_lims[1])))

with open(file_path, 'rb') as f:
    stability_results = pickle.load(f)

stable_timefraction_max = stability_results[0][:, :, 0]

stability_hist = ROOT.TH2D('stability', 'stability', 2020, -10, 2010, 2020, -10, 2010)
for x in range(0, 2000):
    for y in range(0, 2000):
        stability_hist.SetBinContent(x+10, 2010-y, stable_timefraction_max[x, y])

ROOT.gStyle.SetOptStat(0)  # no statistics box on canvas
ROOT.gStyle.SetPalette(60)

DrawHeatMap(stability_hist, 0.1, 1,
            os.path.join('ColorStability_time{:06d}to{:06d}.png'.format(int(t_lims[0]), int(t_lims[1]))), # 'rectangle_0.0-2000.2000', 
            ';pixel X; pixel Y;dominance of main color', False)

pixelstability_distrib = ROOT.TH1D('pixelstab', 'pixelstab', 200, 0, 1)
pixelstability_distrib.SetTitle(';color stability in time;number of pixels')
for x in range(0, 2000):
    for y in range(0, 2000):
        pixelstability_distrib.Fill(stable_timefraction_max[x, y])
        if stable_timefraction_max[x, y] < 0.13:
            print("stability=", stable_timefraction_max[x, y], " for x,y = ", x, y)

c2 = ROOT.TCanvas('c2', 'c2', 2000, 2000)
c2.SetMargin(0.15, 0.05, 0.1, 0.04)  # left,right,bottom,top
pixelstability_distrib.SetLineWidth(2)
pixelstability_distrib.Draw('hist')

c2.SaveAs(os.path.join('figs', 'ColorStability_PixelDistribution_time{:06d}to{:06d}.pdf'.format(int(t_lims[0]), int(t_lims[1])))) # 'rectangle_0.0-2000.2000', 


used_colors_perpixel = ROOT.TH1D('used_colors_perpixel', 'used_colors_perpixel', 32, 0, 32)
for x in range(0, 2000):
    for y in range(0, 2000):
        used_colors_perpixel.Fill(np.count_nonzero(stability_results[0][x, y, :] > 1e-9))

used_colors_perpixel.SetTitle(';number of used colors;number of pixels')
used_colors_perpixel.SetLineWidth(2)

c3 = ROOT.TCanvas('c3', 'c3', 2000, 2000)
c3.SetMargin(0.12, 0.05, 0.1, 0.05)  # left,right,bottom,top
used_colors_perpixel.Draw('hist')
c3.SaveAs(os.path.join('figs', 'NumberOfUsedColors_PixelDistribution_time{:06d}to{:06d}.pdf'.format(int(t_lims[0]), int(t_lims[1])))) # 'rectangle_0.0-2000.2000', 

