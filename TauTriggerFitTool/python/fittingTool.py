import numpy as np
import ROOT

#Class to fit crystalball_cdf to turn-on curve
class fittingTool():
	
	#p0values = [0.2, 0.8, 1.5, 2.5, 5.]			#Alpha
	p0values = [0.2, 0.8, 1.5, 0.3]			#Alpha

	p1values_lowAlpha = [5., 2., 1.5]			#n
	p1values_highAlpha = [3., 1.03]
								#sigma
	p2values_lowestAlpha = [2.]
	p2values_lowAlpha = [7.0, 10., 2., 1.]
	p2values_highAlpha = [25., 15., 7.0]

	p3values = [-25, -40]					#Mean
	#p3values = [-25]					#Mean


	def __init__(self, f, gEfficiency):
		self.f = f
		self.gEfficiency = gEfficiency
	
		#Set some good, general initial guesses	from which to converge to better starting parameters
		self.param = []
		for p3 in self.p3values:
		    for p0 in self.p0values:
			if p0 > 2.:
			    p2values = self.p2values_highAlpha
			    p1values = self.p1values_highAlpha
			else:
			    p2values = self.p2values_lowAlpha
			    p1values = self.p1values_lowAlpha
			
			for p1 in p1values:
				if p0 < 0.5 and p1 < 3.:
				    p2values = self.p2values_lowestAlpha
				
				for p2 in p2values:
				    self.param.append((p0, p1, p2, p3, 1., 1.))
		
		self.second_step_p0values = None
		self.second_step_p1values = None
		self.second_step_p2values = None
		self.second_step_p3values = None
	
	def getValuesAroundParam(self, param_index):
		param = self.param[param_index]
		self.p0values = np.linspace(max(0.01, param[0]-1.), param[0]+1., 10)	
		self.p1values = np.linspace(max(1.02, param[1]-1.5), param[1]+1.5, 12)
		print self.p1values	
		self.p2values = np.linspace(max(0.01, param[2]-2.5), param[2]+2.5, 10)	
		self.p3values = np.linspace(max(-5., param[3]-10.), param[3]+10., 10)	

	def setFitParam(self, ntry):
                self.f.SetParameter( 0, self.param[ntry][0])
                self.f.SetParameter( 1, self.param[ntry][1])
                self.f.SetParameter( 2, self.param[ntry][2])
                self.f.SetParameter( 3, self.param[ntry][3])
                self.f.SetParameter( 4, self.param[ntry][4])
                self.f.SetParameter( 5, self.param[ntry][5])
                
		self.f.SetParError( 0, 0)
                self.f.SetParError( 1, 0)
                self.f.SetParError( 2, 0)
                self.f.SetParError( 3, 0)
                self.f.SetParError( 4, 0)

                #self.f.SetParLimits( 5, 0., 1.5)
                self.f.FixParameter(5, 1.)
		self.f.SetParLimits(1, 1.01, 99)		

	def testFitSuccess(self, verbose = False):
		minuitstatus = ROOT.gMinuit.fCstatu
		if(minuitstatus != "CONVERGED " != 0 and minuitstatus != "OK        "): 
			if verbose:
				print "  Minimization did not converge! (status_\"", minuitstatus,  "\")"
		       	return False
		else:
			return True

	def releaseParameter5(self, ntry):
		self.f.ReleaseParameter(5)
		fit_result2 = self.gEfficiency.Fit(self.f, 'S')
		
		if int(fit_result2) == 0 and fit_result2.CovMatrixStatus() == 3 and self.testFitSuccess():
			return fit_result2
		else:
			#Rather dirty fix to get back to previous fit
			self.setFitParam(ntry)
			print "Fit parameters:", self.f.GetParameter(0), self.f.GetParameter(1), self.f.GetParameter(2), self.f.GetParameter(3), self.f.GetParameter(4), self.f.GetParameter(5)
			fit_result = self.gEfficiency.Fit(self.f, 'S')
			return fit_result
	
	def performFit(self, second_try = False):
	
		#Initial step, check the rough, general estimates
		initial_step_converged = False
		list_of_chi2 = []

		#ROOT.Math.MinimizerOptions.SetDefaultTolerance(0.0000001)

		#Loop over all combinations
		for ntry in xrange(len(self.param)):
			#Set the parameters
			self.setFitParam(ntry)
			print "Fit parameters:", self.f.GetParameter(0), self.f.GetParameter(1), self.f.GetParameter(2), self.f.GetParameter(3), self.f.GetParameter(4), self.f.GetParameter(5)
			
			#Perform the fit and save the chi2
			fit_result = self.gEfficiency.Fit(self.f, 'S')
			list_of_chi2.append(fit_result.Chi2())

			#Check if the fit was successful and the covariant matrix is accurate
			if int(fit_result) == 0 and fit_result.CovMatrixStatus() == 3 and self.testFitSuccess():       
				initial_step_converged = True
				return self.releaseParameter5(ntry)
			else: print 'Fit Failed, start new try'
		
		if second_try:
			return fit_result
		else:
			#Failsafe if previous step did not work (has never been called but is in here just in case)
			#Try to look in the neighbourhood of initial estimate that gave the best fit
			min_chi2_position = list_of_chi2.index(min(list_of_chi2))
			self.getValuesAroundParam(min_chi2_position)
			self.param = []
			for p3 in self.p3values:
			    for p2 in self.p2values:
			       for p1 in self.p1values:
				   for p0 in self.p0values:
				       self.param.append((p0, p1, p2, p3, 1., 1.))

			return self.performFit(second_try=True)	
	
	
