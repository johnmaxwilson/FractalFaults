# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:58:49 2015

@author: jmwilson
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, os

def lerp(start, end, num):
    slope = (end-start)/(num+1)
    x = np.linspace(0, num, num+1)[:-1]
    return slope*x+start


def fractalize(phis, lams, phif, lamf, interdist = 6.0, beta=2.0, rough = 5.0): #interdist expected in units of km
    
    inv = Geodesic.WGS84.Inverse(phis, lams, phif, lamf)    
    line = Geodesic.WGS84.Line(phis, lams, inv['azi1'])
    
    length = int(inv['s12']/(interdist*1000))+1
    
    
    dists = np.linspace(0.0, inv['s12'], num = length)[1:-1]
    
    
    white = np.array(np.random.normal(loc=0,scale=1,size=length), dtype = float)
    count = np.arange(length)
    freq = np.array(np.fft.rfftfreq(length), dtype=np.float)[1:]
    twhite = np.fft.rfft(white)[1:]
    twalk = twhite*(1./freq**(beta/2.0))
    walk = np.fft.irfft(np.hstack(([0],twalk)), length)
    
    walk = walk - ((walk[-1]-walk[0])/length*count + walk[0]) #connect endpoints
    walk = walk[1:-1]/np.std(walk)*(walkstd*1000)             #scale displacement to given std
    
    
    interlats = []
    interlons = []
    for i in range(len(dists)):
        point = line.Position(dists[i])     
        geod = Geodesic.WGS84.Direct(point['lat2'], point['lon2'], point['azi2']-90, walk[i])
        interlats.append(geod['lat2'])
        interlons.append(geod['lon2'])
    
    newlats = np.hstack((phis, interlats))
    newlons = np.hstack((lams, interlons))
    
    
    return newlats, newlons


if __name__ = "__main__":
    
    beta_param = 2.5
    
    beta_filestr = str(beta_param).replace('.', '-')
    
    
    TRACE_DIR = "/home/jmwilson/VirtQuake/fractals/flat/traces/"
    OUTPUT_DIR= "/home/jmwilson/VirtQuake/fractals/fract"+beta_filestr+"/traces/"
    
    #TRACE_DIR = "/home/jmwilson/VirtQuake/VQModels/UCERF3/Traces/original_faultwise/"
    #OUTPUT_DIR= "/home/jmwilson/VirtQuake/VQModels/UCERF3/Traces/fractalized/beta"+beta_filestr"/"
    
    numcopies = 1
    
    
    
    #input_trace = open("traces/singleFault_trace.txt", "r")
    trace_list = os.listdir(TRACE_DIR)
    
    for trace_name in trace_list:
        input_trace = open(TRACE_DIR+trace_name, "r")
        trace = []
        header = []
        
        # Seperate off comments and fault info in header, trace points in trace
        for line in input_trace:
            traceline = line.split()
            if traceline[0] =="#" or len(traceline) == 3:
                header.append(traceline)
            else:
                trace.append([float(el) for el in traceline])
        
        tracerec_names = "lat, lon, alt, depth, slip_rate, aseismic, rake, dip, lame_mu, lame_lambda"
        
        trace_array = np.array(trace)
        trace_rec = np.core.records.fromarrays(trace_array.T, names=tracerec_names)
        
        
        for j in range(numcopies):
            newtrace = np.zeros(shape=(10,1)) #initialize newtrace with the right shape
            
            #For each pair of sequential trace points, fractalize the coords between them, and linear interpolate all other values
            
            for i in range(len(trace_rec)-1):
                newlats, newlons = tools.fractalize(trace_rec['lat'][i], trace_rec['lon'][i], trace_rec['lat'][i+1], trace_rec['lon'][i+1], interdist=6.0, beta=beta_param, walkstd=5.0)
                
                lerpsize = len(newlats)
                altlerp = lerp(trace_rec['alt'][i], trace_rec['alt'][i+1], lerpsize)
                depthlerp = lerp(trace_rec['depth'][i], trace_rec['depth'][i+1], lerpsize)
                sliplerp = lerp(trace_rec['slip_rate'][i], trace_rec['slip_rate'][i+1], lerpsize)
                aseislerp = lerp(trace_rec['aseismic'][i], trace_rec['aseismic'][i+1], lerpsize)
                rakelerp = lerp(trace_rec['rake'][i], trace_rec['rake'][i+1], lerpsize)
                diplerp = lerp(trace_rec['dip'][i], trace_rec['dip'][i+1], lerpsize)
                mulerp = lerp(trace_rec['lame_mu'][i], trace_rec['lame_mu'][i+1], lerpsize)
                lamblerp = lerp(trace_rec['lame_lambda'][i], trace_rec['lame_lambda'][i+1], lerpsize)
                
                temp_trace = np.vstack((newlats, newlons, altlerp, depthlerp, sliplerp, aseislerp, rakelerp, diplerp, mulerp, lamblerp))
                
            
                newtrace = np.hstack((newtrace, temp_trace))
            
            newtrace = np.hstack((newtrace[...,1:], [[el] for el in trace_array[-1]])) #removing first column of initial zeros from newtrace
            
            
            
            # Printing new trace file with same header
            #output_trace = open("traces/singleFaultFract3-0i_trace.txt", 'w')
            output_trace = open(OUTPUT_DIR+"fract"+beta_filestr+"_"+str(j)+"_trace.txt", 'w')
            header[3][1] = len(newtrace[0])
            for outline in header:
                output_trace.write(' '.join([str(el) for el in outline])+'\n')
                
            for outline in newtrace.transpose():
                output_trace.write(' '.join([str(el) for el in outline])+'\n')
            
            output_trace.close()
        
        
        #beta = 2.0
        #D = 1+(3-beta)/2.0
        
        #plt.close('all')
        #plt.plot(newtrace[1], newtrace[0], '.-')
        #plt.ylim(-0.9,0.9)
        #plt.xlim(-0.05, 1.8)
        #plt.show()