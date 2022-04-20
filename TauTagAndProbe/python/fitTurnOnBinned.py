import argparse
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy
from scipy import interpolate

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, KernelOperator, Matern, ConstantKernel, RBF, RationalQuadratic, \
                                             DotProduct, Exponentiation, Hyperparameter

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()

path_prefix = '' if 'TauTriggerTools' in os.getcwd() else 'TauTriggerTools/'
sys.path.insert(0, path_prefix + 'Common/python')
from RootObjects import Histogram, Graph

parser = argparse.ArgumentParser(description='Fit turn-on curves.')
parser.add_argument('--input', required=True, type=str, help="ROOT file with turn-on curves")
parser.add_argument('--output', required=True, type=str, help="output file prefix")
parser.add_argument('--channels', required=False, type=str, default='etau,mutau,ditau', help="channels to process")
parser.add_argument('--decay-modes', required=False, type=str, default='all,0,1,10,11', help="decay modes to process")
parser.add_argument('--working-points', required=False, type=str,
                    default='VVVLoose,VVLoose,VLoose,Loose,Medium,Tight,VTight,VVTight',
                    help="working points to process")
args = parser.parse_args()


class BinnedRBF(Kernel):
    def __init__(self, bins, y_err, l=1.0, l_bounds=(1e-10, 1e10)):
        self.bins = bins
        self.y_err = y_err
        self.l = l
        self.l_bounds = l_bounds

    def is_stationary(self):
        return False

    @property
    def hyperparameter_l(self):
        return Hyperparameter("l", "numeric", self.l_bounds)

    @staticmethod
    def _UtilityFn1(z):
        return z * math.sqrt(math.pi) * scipy.special.erf(z) + np.exp(-(z ** 2))

    @staticmethod
    def _UtilityFn2(z):
        return z * math.sqrt(math.pi) / 2 * scipy.special.erf(z) + np.exp(-(z ** 2))

    @staticmethod
    def _UtilityFn3(z):
        return z * math.sqrt(math.pi) / 2 * scipy.special.erf(z) - z * np.exp(-(z ** 2))

    def FindXYerror(self, a):
        a = np.minimum(self.bins[-1] - 1e-5, a)
        a = np.maximum(self.bins[0], a)
        idx = np.digitize(a, self.bins)
        return (self.bins[idx] - self.bins[idx-1]) / math.sqrt(12), self.y_err[idx - 1], idx
        #return (a - self.bins[0]) / 2 + 1
        #return np.exp((a - self.bins[0]) / 100)

    def FindMutualXYerror(self, a, b):
        a_x_err, a_y_err, a_idx = self.FindXYerror(a)
        b_x_err, b_y_err, b_idx = self.FindXYerror(b)
        L = self.l * (a_x_err[:, np.newaxis] + b_x_err[np.newaxis, :]) / 2
        err_2d = np.tile(a_y_err, (b.shape[0], 1)).T
        noise = np.where(a_idx[:, np.newaxis] == b_idx[np.newaxis, :], err_2d ** 2, 0)
        return L, noise

    def NormDelta(self, a, b, L):
        return (a[:, np.newaxis] - b[np.newaxis, :]) / L

    def k_ff(self, X, Y):
        u = X[:, 0]
        up = Y[:, 0]
        L, noise = self.FindMutualXYerror(u, up)
        return np.exp(- self.NormDelta(u, up, L) ** 2 ) + noise

    def k_FF(self, X, Y):
        s = X[:, 0]
        t = X[:, 1]
        sp = Y[:, 0]
        tp = Y[:, 1]
        fn = BinnedRBF._UtilityFn1
        L, noise = self.FindMutualXYerror(s, sp)
        d = [ self.NormDelta(t, sp, L), self.NormDelta(s, tp, L),
              self.NormDelta(t, tp, L), self.NormDelta(s, sp, L) ]
        sum = fn(d[0]) + fn(d[1]) - fn(d[2]) - fn(d[3])
        norm = (t - s)[:, np.newaxis] * (tp - sp)[np.newaxis, :]
        return ((L ** 2) / 2 * sum + noise) / norm

    def k_Ff(self, X, Y):
        s = X[:, 0]
        t = X[:, 1]
        tp = Y[:, 0]
        fn = scipy.special.erf
        L, noise = self.FindMutualXYerror(s, tp)
        d = [ self.NormDelta(t, tp, L), self.NormDelta(s, tp, L) ]
        sum = fn(d[0]) - fn(d[1])
        #norm = np.tile(t - s, (tp.shape[0], 1)).T
        norm = (t - s)[:, np.newaxis]
        print(noise.shape, norm.shape, L.shape)
        return (L * math.sqrt(math.pi) / 2 * sum + noise) / norm

    def k_ff_grad_l(self, X, Y):
        u = X[:, 0]
        up = Y[:, 0]
        L, noise = self.FindMutualXYerror(u, up)
        delta2 = self.NormDelta(u, up, L) ** 2
        return 2 * delta2 / L * np.exp(-delta2)

    def k_FF_grad_l(self, X, Y):
        s = X[:, 0]
        t = X[:, 1]
        sp = Y[:, 0]
        tp = Y[:, 1]
        fn = BinnedRBF._UtilityFn2
        L, noise = self.FindMutualXYerror(s, sp)
        d = [ self.NormDelta(t, sp, L), self.NormDelta(s, tp, L),
              self.NormDelta(t, tp, L), self.NormDelta(s, sp, L) ]
        sum = fn(d[0]) + fn(d[1]) - fn(d[2]) - fn(d[3])
        norm = (t - s)[:, np.newaxis] * (tp - sp)[np.newaxis, :]
        return L * sum / norm

    def k_Ff_grad_l(self, X, Y):
        s = X[:, 0]
        t = X[:, 1]
        tp = Y[:, 0]
        fn = BinnedRBF._UtilityFn3
        L, noise = self.FindMutualXYerror(s, tp)
        sum = fn(self.NormDelta(t, tp, L)) + fn(-self.NormDelta(s, tp, L))
        norm = (t - s)[:, np.newaxis]
        return sum / norm

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            x = X
            y = X
        else:
            x = X
            y = Y
        ndim_x = x.shape[1]
        ndim_y = y.shape[1]

        transpose = False
        if ndim_x == 1 and ndim_y == 1:
            K_fn = self.k_ff
            K_grad_fn = self.k_ff_grad_l
        elif ndim_x == 2 and ndim_y == 2:
            K_fn = self.k_FF
            K_grad_fn = self.k_FF_grad_l
        elif (ndim_x == 2 and ndim_y == 1) or (ndim_x == 1 and ndim_y == 2):
            if ndim_x == 1 and ndim_y == 2:
                x, y = y, x
                transpose = True
            K_fn = self.k_Ff
            K_grad_fn = self.k_Ff_grad_l
        else:
            raise RuntimeError("BinnedRBF: mode ndim_x = {} and ndim_y = {} is not supported.".format(ndim_x, ndim_y))

        K = K_fn(x, y)
        if transpose:
            K = K.T
        if eval_gradient:
            if self.hyperparameter_l.fixed:
                K_grad = np.empty((K.shape[0], K.shape[1], 0))
            else:
                K_grad = K_grad_fn(x, y)
                if transpose:
                    K_grad = K_grad.T
                K_grad = K_grad[:, :, np.newaxis]
            return K, K_grad
        return K

    def diag(self, X):
        return np.copy(np.diag(self(X, X)))

class FitResults:
    def __init__(self, eff, x_pred, pred_bin_width):
        N = eff.x.shape[0]
        bins = np.zeros((N, 2))
        bins[:, 0] = eff.x - eff.x_error_low
        bins[:, 1] = eff.x + eff.x_error_high
        bins_1d = np.append(bins[:, 0], [ bins[-1, 1] ])

        bin_width = bins[:, 1] - bins[:, 0]
        yerr = np.maximum(eff.y_error_low, eff.y_error_high)

        #kernel = ConstantKernel() * BinnedRBF(bins_1d, yerr)
        kernel = BinnedRBF(bins_1d, yerr)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp.fit(bins, eff.y)

        pred_bins = np.zeros((len(x_pred), 2))
        pred_bins[:, 0] = x_pred - pred_bin_width / 2
        pred_bins[:, 1] = x_pred + pred_bin_width / 2
        self.y_pred, self.sigma_pred = self.gp.predict(np.atleast_2d(x_pred).T, return_std=True)
        #self.y_pred, self.sigma_pred = self.gp.predict(pred_bins, return_std=True)
        #self.y_pred *= pred_bin_width
        #self.sigma_pred *= pred_bin_width

channels = args.channels.split(',')
decay_modes = args.decay_modes.split(',')
working_points = args.working_points.split(',')

file = ROOT.TFile(args.input, 'READ')
output_file = ROOT.TFile('{}.root'.format(args.output), 'RECREATE', '', ROOT.RCompressionSetting.EDefaults.kUseSmallest)

for channel in channels:
    print('Processing {}'.format(channel))
    with PdfPages('{}_{}.pdf'.format(args.output, channel)) as pdf:
        for wp in working_points:
            for dm in decay_modes:
                dm_label = '_dm{}'.format(dm) if dm != 'all' else ''
                name_pattern = '{{}}_{}_{}{}_fit_eff'.format(channel, wp, dm_label)
                dm_label = '_dm'+ dm if len(dm) > 0 else ''
                eff_data_root = file.Get(name_pattern.format('data'))
                eff_mc_root = file.Get(name_pattern.format('mc'))
                eff_data = Graph(root_graph=eff_data_root)
                eff_mc = Graph(root_graph=eff_mc_root)
                pred_step = 0.1
                #x_low = min(eff_data.x[0] - eff_data.x_error_low[0], eff_mc.x[0] - eff_mc.x_error_low[0])
                #x_high = max(eff_data.x[-1] + eff_data.x_error_high[-1], eff_mc.x[-1] + eff_mc.x_error_high[-1])
                x_low, x_high = 20, 1000
                x_pred = np.arange(x_low, x_high + pred_step / 2, pred_step)

                eff_data_fitted = FitResults(eff_data, x_pred, pred_step)
                eff_mc_fitted = FitResults(eff_mc, x_pred, pred_step)

                fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                                           gridspec_kw = {'height_ratios':[2, 1]})
                mc_color = 'g'
                data_color = 'k'
                trans = 0.3

                ax.errorbar(eff_mc.x, eff_mc.y, xerr=(eff_mc.x_error_low, eff_mc.x_error_high),
                            yerr=(eff_mc.y_error_low, eff_mc.y_error_high), fmt=mc_color+'.', markersize=5)
                ax.errorbar(eff_data.x, eff_data.y, xerr=(eff_data.x_error_low, eff_data.x_error_high),
                            yerr=(eff_data.y_error_low, eff_data.y_error_high), fmt=data_color+'.', markersize=5)
                ax.plot(x_pred, eff_mc_fitted.y_pred, mc_color+'--')
                ax.fill(np.concatenate([x_pred, x_pred[::-1]]),
                        np.concatenate([eff_mc_fitted.y_pred - eff_mc_fitted.sigma_pred,
                                       (eff_mc_fitted.y_pred + eff_mc_fitted.sigma_pred)[::-1]]),
                        alpha=trans, fc=mc_color, ec='None')
                ax.plot(x_pred, eff_data_fitted.y_pred, data_color+'--')
                ax.fill(np.concatenate([x_pred, x_pred[::-1]]),
                        np.concatenate([eff_data_fitted.y_pred - eff_data_fitted.sigma_pred,
                                       (eff_data_fitted.y_pred + eff_data_fitted.sigma_pred)[::-1]]),
                        alpha=trans, fc=data_color, ec='None')

                title = "Turn-ons for {} trigger with {} DeepTau VSjet".format(channel, wp)
                if dm != 'all':
                    title += " for DM={}".format(dm)
                else:
                    title += " for all DMs"
                plt.title(title, fontsize=16)
                plt.xlabel("$p_T$ (GeV)", fontsize=12)
                plt.ylabel("Efficiency", fontsize=12)
                plt.ylim([ 0., 1.1 ])
                plt.xlim([ 20, min(200, plt.xlim()[1]) ])
                pdf.savefig(bbox_inches='tight')
                plt.close()

                out_name_pattern = '{{}}_{}_{}{}_{{}}'.format(channel, wp, dm_label)
                output_file.WriteTObject(eff_data_root, out_name_pattern.format('data', 'eff'), 'Overwrite')
                output_file.WriteTObject(eff_mc_root, out_name_pattern.format('mc', 'eff'), 'Overwrite')
                eff_data_fitted_hist = Histogram.CreateTH1(eff_data_fitted.y_pred, [x_low, x_high],
                                                           eff_data_fitted.sigma_pred, fixed_step=True)
                eff_mc_fitted_hist = Histogram.CreateTH1(eff_mc_fitted.y_pred, [x_low, x_high],
                                                         eff_mc_fitted.sigma_pred, fixed_step=True)
                sf_fitted_hist = eff_data_fitted_hist.Clone()
                sf_fitted_hist.Divide(eff_mc_fitted_hist)
                output_file.WriteTObject(eff_data_fitted_hist, out_name_pattern.format('data', 'fitted'), 'Overwrite')
                output_file.WriteTObject(eff_mc_fitted_hist, out_name_pattern.format('mc', 'fitted'), 'Overwrite')
                output_file.WriteTObject(sf_fitted_hist, out_name_pattern.format('sf', 'fitted'), 'Overwrite')

output_file.Close()
print('All done.')
