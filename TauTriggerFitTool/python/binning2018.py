from array import array
from math import sqrt
import numpy as np

class getbinning2018:

	def __init__(self):
		pass

	def getBinning(self):
		edgesdiTau = []
		for i in range( 20, 50, 2) :
			edgesdiTau.append( float(i) )
		for i in range( 50, 75, 5 ) :
			edgesdiTau.append( float(i) )
		for i in range( 75, 100, 25) :
   			edgesdiTau.append( float(i) )
		for i in range(100, 150, 50 ) :
			edgesdiTau.append( float(i) )
		for i in range(150, 500, 300 ) :
 	  	 	edgesdiTau.append( float(i) )
 	  	 
 	  	edgesMuTau =[]
		for i in range( 20, 50, 2) :
			edgesMuTau.append( float(i) )
		for i in range( 50, 75, 5 ) :
			edgesMuTau.append( float(i) )
		for i in range( 75, 100, 25) :
   			edgesMuTau.append( float(i) )
		for i in range(100, 150, 50 ) :
			edgesMuTau.append( float(i) )
		for i in range(150, 500, 300 ) :
 	  	 	edgesMuTau.append( float(i) )
 	  	 	
 	  	edgesETau =[]
		for i in range( 20, 24, 1) :
			edgesETau.append( float(i) )
		for i in range( 24, 30, 3) :
			edgesETau.append( float(i) )
		for i in range( 30, 55, 5 ) :
			edgesETau.append( float(i) )
		for i in range( 55, 70, 15 ) :
			edgesETau.append( float(i) )
		for i in range( 70, 90, 20) :
			edgesETau.append( float(i) )
		for i in range(90, 160, 50 ) :
			edgesETau.append( float(i) )
		for i in range(160, 600, 350 ) :
			edgesETau.append( float(i) )

		#ref_bins = np.concatenate((np.arange(20, 100, 1), np.arange(100, 200, 5), np.arange(200, 301, 10)))
		ref_bins = np.concatenate((np.arange(20, 50, 1), np.arange(50, 100, 2), np.arange(100, 150, 5),
                         np.arange(150, 200, 10), np.arange(200, 301, 20)))
		edgesdiTau = ref_bins.tolist()
		edgesMuTau = ref_bins.tolist()
		edgesETau = ref_bins.tolist()
		#edgesdiTau = np.arange(20, 501, 1).tolist()
		#edgesMuTau = np.arange(20, 501, 1).tolist()
		#edgesETau = np.arange(20, 501, 1).tolist()
			
		edgesdict ={"ditau":edgesdiTau, "mutau":edgesMuTau, "etau":edgesETau}
 	 	return edgesdict
 	 	
 	def getBinningDM(self):
 	 	edgesDiTauDM0MC = []
 	 	for i in range( 20, 40, 2) : 
			edgesDiTauDM0MC.append( float(i) )
		for i in range( 40, 50, 5) : 
			edgesDiTauDM0MC.append( float(i) )
		for i in range( 50, 65, 15 ) : 
			edgesDiTauDM0MC.append( float(i) )
		for i in range( 65, 80, 15) : 
			edgesDiTauDM0MC.append( float(i) )
		for i in range(80, 110, 30) :
			edgesDiTauDM0MC.append( float(i) )
		for i in range(110, 500, 300 ) :  
			edgesDiTauDM0MC.append( float(i) )
			
		edgesDiTauDM0Data = []
 	 	for i in range( 20, 40, 2) :
			edgesDiTauDM0Data.append( float(i) )
		for i in range( 40, 55, 5) :
			edgesDiTauDM0Data.append( float(i) )
		for i in range( 55, 65, 10 ) :
			edgesDiTauDM0Data.append( float(i) )
		for i in range( 65, 85, 20) :
			edgesDiTauDM0Data.append( float(i) )
		for i in range(85, 115, 30) :
			edgesDiTauDM0Data.append( float(i) )
		for i in range(115, 500, 300 ) :
			edgesDiTauDM0Data.append( float(i) )
		
		edgesDiTauDM1 = []	
		for i in range( 20, 30, 2) :
			edgesDiTauDM1.append( float(i) )
		for i in range( 30, 55, 5) :
			edgesDiTauDM1.append( float(i) )
		for i in range( 55, 65, 10 ) :
			edgesDiTauDM1.append( float(i) )
		for i in range( 65, 85, 20) :
			edgesDiTauDM1.append( float(i) )
		for i in range(85, 110, 25 ) :
			edgesDiTauDM1.append( float(i) )
		for i in range(110, 500, 300 ) :
			edgesDiTauDM1.append( float(i) )
				
		edgesDiTauDM10Data = []	
		for i in range(20, 30, 2) :
			edgesDiTauDM10Data.append( float(i) ) 
		for i in range(30, 50, 5) :
			edgesDiTauDM10Data.append( float(i) )
		for i in range(50, 70, 10 ) :
			edgesDiTauDM10Data.append( float(i) )
		for i in range(70, 85, 15 ) :
			edgesDiTauDM10Data.append( float(i) )
		for i in range(85, 115, 30 ) :
			edgesDiTauDM10Data.append( float(i) )
		for i in range(115, 500, 300 ) :
			edgesDiTauDM10Data.append( float(i) )
			
		edgesDiTauDM10MC = []				
		for i in range(20, 30, 2) :
			edgesDiTauDM10MC.append( float(i) )
		for i in range(30, 45, 5) :
			edgesDiTauDM10MC.append( float(i) )
		for i in range(45, 55, 10) :
			edgesDiTauDM10MC.append( float(i) )
		for i in range(55, 85, 15 ) :
			edgesDiTauDM10MC.append( float(i) )
		for i in range(85, 125, 40 ) :
			edgesDiTauDM10MC.append( float(i) )
		for i in range(125, 500, 300 ) :
			edgesDiTauDM10MC.append( float(i) )
		edgesDiTauDM0 = {"DATA":edgesDiTauDM0Data, "MC": edgesDiTauDM0MC}
		edgesDiTauDM1 = {"DATA":edgesDiTauDM1, "MC": edgesDiTauDM1}
                edgesDiTauDM10 = {"DATA":edgesDiTauDM10Data, "MC": edgesDiTauDM10MC}

                edgesDiTauDM = {"dm0": edgesDiTauDM0, "dm1": edgesDiTauDM1, "dm10": edgesDiTauDM10}
                
                edgesMuTauDM0Data = []
               	for i in range(20, 24, 1) :
			edgesMuTauDM0Data.append( float(i) )
		for i in range(24, 30, 2) :
			edgesMuTauDM0Data.append( float(i) )
		for i in range( 30, 40, 5) :
			edgesMuTauDM0Data.append( float(i) )
		for i in range( 40, 50, 25 ) :
			edgesMuTauDM0Data.append( float(i) )
		for i in range( 50, 65, 15) :
			edgesMuTauDM0Data.append( float(i) )
		for i in range(65, 105, 40 ) :
			edgesMuTauDM0Data.append( float(i) )
		for i in range(105, 500, 300 ) :
			edgesMuTauDM0Data.append( float(i) )
			
		edgesMuTauDM0MC = []
               	for i in range(20, 22, 2) :
			edgesMuTauDM0MC.append( float(i) )
		for i in range(22, 30, 2) :
			edgesMuTauDM0MC.append( float(i) )
		for i in range( 30, 35, 5) :
			edgesMuTauDM0MC.append( float(i) )
		for i in range( 35, 45, 10 ) :
			edgesMuTauDM0MC.append( float(i) )
		for i in range( 45, 65, 20) :
			edgesMuTauDM0MC.append( float(i) )
		for i in range(65, 95, 35 ) :
			edgesMuTauDM0MC.append( float(i) )
		for i in range(95, 500, 300 ) :
			edgesMuTauDM0MC.append( float(i) )
		
		
		edgesMuTauDM1 = []
		for i in range(20, 24, 1) :
			edgesMuTauDM1.append( float(i) )
		for i in range(24, 30, 2) :
			edgesMuTauDM1.append( float(i) )
		for i in range( 30, 40, 5) :
			edgesMuTauDM1.append( float(i) )
		for i in range( 40, 60, 10 ) :
			edgesMuTauDM1.append( float(i) )
		for i in range( 60, 75, 15) :
			edgesMuTauDM1.append( float(i) )
		for i in range(75, 110, 35 ) :
			edgesMuTauDM1.append( float(i) )
		for i in range(110, 500, 300 ) :
			edgesMuTauDM1.append( float(i) )
			
		edgesMuTauDM10 = []
		for i in range(20, 24, 2) :
			edgesMuTauDM10.append( float(i) )
		for i in range(24, 30, 2) :
			edgesMuTauDM10.append( float(i) )
		for i in range(30, 40, 5 ) :
			edgesMuTauDM10.append( float(i) )
		for i in range(40, 58, 9 ) :
			edgesMuTauDM10.append( float(i) )
		for i in range(58, 80, 22 ) :
			edgesMuTauDM10.append( float(i) )
		for i in range(80, 115, 35) :
			edgesMuTauDM10.append( float(i) )
		for i in range(115,  500, 300 ) :
			edgesMuTauDM10.append( float(i) )

                edgesMuTauDM0 = {"DATA":edgesMuTauDM0Data, "MC": edgesMuTauDM0MC}
                edgesMuTauDM1 = {"DATA":edgesMuTauDM1, "MC": edgesMuTauDM1}
                edgesMuTauDM10 = {"DATA":edgesMuTauDM10, "MC": edgesMuTauDM10}

                edgesMuTauDM = {"dm0": edgesMuTauDM0, "dm1": edgesMuTauDM1, "dm10": edgesMuTauDM10}
                
                edgesETauDM0Data = []
                for i in range( 20, 22, 1) :
                        edgesETauDM0Data.append( float(i) )
                for i in range( 22, 24, 2) :
                        edgesETauDM0Data.append( float(i) )
                for i in range( 24, 30, 3) :
                        edgesETauDM0Data.append( float(i) )
		for i in range( 30, 45, 5) :
			edgesETauDM0Data.append( float(i) )
		for i in range( 45, 55, 10 ) :
			edgesETauDM0Data.append( float(i) )
		for i in range( 55, 65, 10 ) :
			edgesETauDM0Data.append( float(i) )
		for i in range( 65, 100, 35) :
			edgesETauDM0Data.append( float(i) )
		for i in range(100, 500, 300 ) :
			edgesETauDM0Data.append( float(i) ) 
			
		edgesETauDM0MC = []
                for i in range( 20, 24, 2) :
                        edgesETauDM0MC.append( float(i) )
                for i in range( 24, 30, 3) :
                        edgesETauDM0MC.append( float(i) )
		for i in range( 30, 50, 5) :
			edgesETauDM0MC.append( float(i) )
		for i in range( 50, 60, 10 ) :
			edgesETauDM0MC.append( float(i) )
		for i in range( 60, 75, 15) :
			edgesETauDM0MC.append( float(i) )
		for i in range(75, 95, 20 ) :
			edgesETauDM0MC.append( float(i) )
		for i in range(95, 500, 300 ) :
			edgesETauDM0MC.append( float(i) ) 

                edgesETauDM1 = []
                for i in range( 20, 30, 2) :
                        edgesETauDM1.append( float(i) )
                for i in range( 30, 40, 5) :
                        edgesETauDM1.append( float(i) )
                for i in range( 40, 55, 5 ) :
                        edgesETauDM1.append( float(i) )
                for i in range( 55, 75, 20) :
                        edgesETauDM1.append( float(i) )
		for i in range(75, 95, 20 ) :
			edgesETauDM1.append( float(i) )
		for i in range(95, 500, 350 ) :
			edgesETauDM1.append( float(i) )

                edgesETauDM10Data = []
                for i in range( 20, 22, 1) :
                        edgesETauDM10Data.append( float(i) )
                for i in range( 22, 30, 2) :
                        edgesETauDM10Data.append( float(i) )
                for i in range( 30, 55, 5 ) :
                        edgesETauDM10Data.append( float(i) )
		for i in range( 55, 75, 20 ) :
			edgesETauDM10Data.append( float(i) )
		for i in range(75, 100, 25 ) :
			edgesETauDM10Data.append( float(i) )
		for i in range(100, 600, 300 ) :
			edgesETauDM10Data.append( float(i) )
			
                edgesETauDM10MC = []
                for i in range( 20, 22, 1) :
                        edgesETauDM10MC.append( float(i) )
                for i in range( 22, 30, 2) :
                        edgesETauDM10MC.append( float(i) )
                for i in range( 30, 55, 5 ) :
                        edgesETauDM10MC.append( float(i) )
		for i in range( 55, 75, 20 ) :
			edgesETauDM10MC.append( float(i) )
		for i in range(75, 100, 25 ) :
			edgesETauDM10MC.append( float(i) )
		for i in range(100, 600, 300 ) :
			edgesETauDM10MC.append( float(i) )

                edgesETauDM0 = {"DATA":edgesETauDM0Data, "MC": edgesETauDM0MC}
                edgesETauDM1 = {"DATA":edgesETauDM1, "MC": edgesETauDM1}
                edgesETauDM10 = {"DATA":edgesETauDM10Data, "MC": edgesETauDM10MC}
                
                edgesETauDM = {"dm0": edgesETauDM0, "dm1": edgesETauDM1, "dm10": edgesETauDM10} 
                edgesDM =  {"ditau": edgesDiTauDM, "mutau": edgesMuTauDM, "etau": edgesETauDM}
                return edgesDM
        

