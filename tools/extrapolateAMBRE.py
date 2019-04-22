import numpy as np
import pandas as pd
from astropy.io import ascii, fits
from astropy.table import Table
from matplotlib import pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from scipy.optimize import curve_fit, OptimizeWarning
from scipy import constants
import os
import os.path
from matplotlib.ticker import AutoMinorLocator
import argparse
import datetime

class AmbreModel:
    """ Extrapolate a spectrum at > 1.2 micron from a model spectrum in the 
        AMBRE library. This class is able to smooth a spectrum with Gaussian 
        kernek and do binning it. Fitting with an exponential function is 
        performed for an extrapolation. 
        
        Note: 
        - Model data with NaN values is not supported.
        - A channel scale of the original spectrum is assumed to be 0.001 nm.
    """

    def __init__(self, data):
        self.data = data

    @staticmethod
    def fromFile(modelPath):
        """ Read a model file.
            The unit of wavelength is converted from \AA to nm. """
        modelData = ascii.read(modelPath, format='no_header', names=['wavelength', 'normalized_flux', 'flux'])
        model = modelData.to_pandas()
        model['flux'] = model['flux']/1e6 ## scale down to 1e-6 erg/s/cm^2/A
        model['wavelength'] = model['wavelength']/10.
        return AmbreModel(model)

    def gaussSmooth(self, gaussSmoothWidth, mode):
        """ Smooth a spectrum with the Gaussian kernel of FWHM (`gaussSmoothWidth` 
            in nm)
            Two options (`mode`): 'original' uses the original data, 'binned'
            uses the binned data after `doBinningPix`.  
        """
        if mode == 'original':
            model = self.data
            delta = 0.001 ## channel scale 0.001 nm/pix
            gaussSigma = gaussSmoothWidth/delta/(2.*np.sqrt(2.*np.log(2.)))
            gaussKernel = Gaussian1DKernel(stddev=gaussSigma)
            gaussSmoothedFlux = convolve(model['flux'], gaussKernel, normalize_kernel=True, 
                                         boundary='fill', fill_value=np.nan)
        if mode == 'binned':
            model = self.resample
            delta = model['wavelength'][1] - model['wavelength'][0] ## channel scale in nm/pix
            gaussSigma = gaussSmoothWidth/delta/(2.*np.sqrt(2.*np.log(2.)))
            gaussKernel = Gaussian1DKernel(stddev=gaussSigma)
            gaussSmoothedFlux = convolve(model['binned_flux'], gaussKernel, normalize_kernel=True, 
                                         boundary='fill', fill_value=np.nan)
        if (mode != 'original') and (mode != 'binned'):
            print('ERROR: Use either `original` or `binned` in `mode`.')

        model['gauss_smoothed_flux'] = gaussSmoothedFlux
        maskBool = np.isnan(gaussSmoothedFlux)
        if 'mask' in model.columns.values:
            maskBool = np.maximum(model['mask'], maskBool)
        model['mask'] = maskBool
        return self

    def maskPaschen(self, paWidth, mode):
        """ Mask Paschen absoroptions from 6-3 to 10-3 transitions.
            `mask` column is set to be True for masked channels
            Two options (`mode`): 'original' uses the original data, 'binned'
            uses the binned data after `doBinningPix`.  
        """
        ## Wavelengths of the 10-3, 9-3, 8-3, 7-3, and 6-3 transitions in Paschen
        paWavelength = np.array([901.74, 923.15, 954.86, 1005.21, 1094.11])   
        paWidthMicron = paWidth/2.

        if mode == 'original':
            model = self.data
        if mode == 'binned':
            model = self.resample
        if (mode != 'original') and (mode != 'binned'):
            print('ERROR: Use either `original` or `binned` in `mode`.')

        maskBool = (
                   ((model['wavelength'] > paWavelength[0] - paWidthMicron) & 
                   (model['wavelength'] < paWavelength[0] + paWidthMicron)) |
                   ((model['wavelength'] > paWavelength[1] - paWidthMicron) & 
                   (model['wavelength'] < paWavelength[1] + paWidthMicron)) |
                   ((model['wavelength'] > paWavelength[2] - paWidthMicron) & 
                   (model['wavelength'] < paWavelength[2] + paWidthMicron)) |
                   ((model['wavelength'] > paWavelength[3] - paWidthMicron) & 
                   (model['wavelength'] < paWavelength[3] + paWidthMicron)) |
                   ((model['wavelength'] > paWavelength[4] - paWidthMicron) & 
                   (model['wavelength'] < paWavelength[4] + paWidthMicron))
                   )
        if 'mask' in model.columns.values:
            maskBool = np.maximum(model['mask'], maskBool)
        model['mask'] = maskBool
        return self

    def exponentialFit(self, fitRange, mode):
        """ Fit the model with an exponential function. Fitting range is
            specified by `fitRange`.
            Two options (`mode`): 'original' uses the original data, 'binned'
            uses the binned data after `doBinningPix`.  
        """
        if mode == 'original':
            model = self.data
        if mode == 'binned':
            model = self.resample
        if (mode != 'original') and (mode != 'binned'):
            print('ERROR: Use either `original` or `binned` in `mode`.')

        ## set `mask` True for channels out of the range for fitting
        cutBool = ~((model['wavelength'] >= fitRange[0]) & (model['wavelength'] <= fitRange[1]))
        if 'mask' in model.columns.values:
            cutBool = np.maximum(model['mask'], cutBool)
        fitMask = cutBool

        if 'mask' in model.columns.values:
            try:
                fitParameter, covariance = curve_fit(exponenialFunc, 
                                                     nmToMicron(model[~fitMask]['wavelength']), 
                                                     model[~fitMask]['gauss_smoothed_flux'], maxfev=5000)
            except (RuntimeError, OptimizeWarning) as error:
                print(error)
                fitParameter = np.array([np.nan, np.nan, np.nan])
                fitParameterError = np.array([np.nan, np.nan, np.nan])
                fitSTD = np.nan
                correlationCoefficient = np.nan
            else:
                fitParameter[1] = nmToMicron(fitParameter[1])
                fitParameterError = np.sqrt(np.diag(covariance))
                bestFit = exponenialFunc(model[~fitMask]['wavelength'], 
                                         fitParameter[0], fitParameter[1], fitParameter[2])
                ## standard deviation of data around the best fit
                fitSTD = np.std((model[~fitMask]['gauss_smoothed_flux'] - bestFit), ddof=4)
                ## correlation coefficient
                correlationCoefficient = np.corrcoef(model[~fitMask]['gauss_smoothed_flux'], bestFit)[1,0]
        return fitParameter, fitParameterError, fitSTD, correlationCoefficient

    def doBinningPix(self, boxSmoothWidth):
        """ Binning a spectrum with the boxcar kernel of a width = 
            `boxSmoothWidth` in nm. The number of channels is reduced.
        """
        model = self.data
        delta = 0.001 ## channel scale 0.001 nm/pix
        binnedChannel = int(boxSmoothWidth/delta)
        indexResample = np.arange(int(len(model['wavelength'])//binnedChannel))

        resampleWavelength = np.zeros_like(indexResample, dtype='float64')
        resampleFlux = np.zeros_like(indexResample, dtype='float64')
        resampleMask = np.zeros_like(indexResample, dtype='bool')

        wavelength = model['wavelength']
        flux = model['flux']
        if 'mask' in model.columns.values:
            mask = model['mask']

        for j in indexResample:
            resampleWavelength[j] = (wavelength[j*binnedChannel] + wavelength[(j+1)*binnedChannel - 1])/2.
            resampleFlux[j] = np.mean(flux[j*binnedChannel:(j+1)*binnedChannel-1+1])
            if 'mask' in model.columns.values: 
                resampleMask[j] = any(mask[j*n:(j+1)*n-1+1])

        resampleModel = pd.DataFrame({'wavelength': resampleWavelength, 
                                      'binned_flux': resampleFlux})

        if 'mask' in model.columns.values: 
            resampleModel = pd.DataFrame({'wavelength': resampleWavelength, 
                                          'binned_flux': resampleFlux,
                                          'mask': resampleMask})
        self.resample = resampleModel
        return self


def exponenialFunc(x, a, b, c):      
    return a*np.exp(-b*x)+c

def nmToMicron(data):
    """ Convert nm to micron """
    return data/1e3

def specExtrapolate(wavelength, flux, delta, fitParameter, extrapolateRange=[0.,1260.000]):
    """ Extrapolate a spectrum using `exponenialFunc` and `fitParameter`.
    """
    if extrapolateRange[0] == 0.:
        ## Get the longest wavelength. Both edges were cutted in `gaussSmooth()`.
        nanData = np.isnan(flux)
        longestId = wavelength[~nanData].idxmax()
        extrapolateRange[0] = wavelength[longestId]
    step = int((extrapolateRange[1] - extrapolateRange[0]) / delta + 1.)
    extrapolationX = np.linspace(extrapolateRange[0]+delta, extrapolateRange[1], step, 
                                 endpoint=True, dtype='float64')
    extrapolationY = exponenialFunc(extrapolationX, fitParameter[0], fitParameter[1], fitParameter[2])
    extrapolation = pd.DataFrame({'wavelength': extrapolationX, 'flux': extrapolationY})
    return extrapolation

def doProcessing(dataPath, i, saveDir, fitRange, boxSmoothWidth, gaussSmoothWidth, paschenWidth):
    """ Channel binning is performed for a input spectrum. Then the spectrum is
        smoothed with the Gaussian kernel with 2.4A resolution and is fitted 
        with an exponential function. The extrapolated spectrum is generated 
        from the fit curve.
        Processed spectrum is saved into .fits and plotted in .png.
    """
    baseName = os.path.basename(dataPath)
    saveName = os.path.basename(dataPath).replace(':', '_')
    summaryFile = saveDir+'summary.dat'  ## summary file for output
    xStart = fitRange[0]   ## the minimum wavelength of the fitting range
    xEnd = fitRange[1]     ## the maximum wvelength of the fitting range
    extrapolateEnd = 1300.000 ## the maximum wavelength to be extrapoted (nm)

    ## Processing
    model = AmbreModel.fromFile(dataPath)
    model.doBinningPix(boxSmoothWidth)
    model.gaussSmooth(gaussSmoothWidth, 'binned') ## mode = data or resample
    model.maskPaschen(paschenWidth, 'binned')
    fitParameter, fitParameterError, fitSTD, correlationCoefficien = model.exponentialFit([xStart, xEnd], 
                                                                                          'binned')
    originalData = model.data
    resampleData = model.resample
    fittingWavelength = ((~resampleData['mask']) & 
                        (resampleData['wavelength'] >= xStart) & 
                        (resampleData['wavelength'] <= xEnd))
    if all(np.isfinite(fitParameter)):
        bestFit = exponenialFunc(resampleData[fittingWavelength]['wavelength'], 
                                 fitParameter[0], fitParameter[1], fitParameter[2])
        extrapolation = specExtrapolate(resampleData['wavelength'], resampleData['binned_flux'], 
                                        boxSmoothWidth, fitParameter, [0., extrapolateEnd])
        extrapolation = extrapolation.rename(columns={'flux': 'binned_flux'}) 

    ## Plot the spectrum 
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(originalData['wavelength'], originalData['flux'], '-k', lw=0.5, alpha=0.2, label='Original data')
    ax1.plot(resampleData['wavelength'], resampleData['binned_flux'], '-k', lw=0.5, alpha=0.7, 
             label='Binnded data')
    ax1.plot(resampleData['wavelength'], resampleData['gauss_smoothed_flux'], '-b', lw=1, 
             label='Smoothed data')
    ax1.plot(resampleData[fittingWavelength]['wavelength'], 
             resampleData[fittingWavelength]['gauss_smoothed_flux'], 
             '-y', lw=1, label='Used data for fitting')
    plt.xlim(280.0, 1330.0)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_title(saveName)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Flux density $(10^{-6}$ erg/s/cm$^2$/$\AA$)')    
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(originalData['wavelength'], originalData['flux'], '-k', lw=0.5, alpha=0.2, label='Original data')
    ax2.plot(resampleData['wavelength'], resampleData['binned_flux'], '-k', lw=0.5, alpha=0.7, 
             label='Binnded data')
    ax2.plot(resampleData['wavelength'], resampleData['gauss_smoothed_flux'], '-b', lw=1, 
             label='Smoothed data')
    ax2.plot(resampleData[fittingWavelength]['wavelength'], 
             resampleData[fittingWavelength]['gauss_smoothed_flux'], 
             '-y', lw=1, label='Used data for fitting')
    if all(np.isfinite(fitParameter)):
        ax1.plot(resampleData[fittingWavelength]['wavelength'], bestFit, '-c', lw=1, label='Best fit line')
        ax1.plot(extrapolation['wavelength'], extrapolation['binned_flux'], '-r', lw=1.0,
                 label='Extrapolation') 
        ax2.plot(resampleData[fittingWavelength]['wavelength'], bestFit, '-c', lw=1, label='Best fit line')
        ax2.plot(extrapolation['wavelength'], extrapolation['binned_flux'], '-r', lw=1.0, 
                 label='Extrapolation')
    plt.xlim(xStart, 1310.0)
    plt.ylim(originalData[(originalData['wavelength'] > xStart) & 
                          (originalData['wavelength'] < xEnd)]['flux'].min()*0.5,
             originalData[(originalData['wavelength'] > xStart) & 
                          (originalData['wavelength'] < xEnd)]['flux'].max()*1.5)
    ax2.xaxis.set_minor_locator(AutoMinorLocator()) 
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Flux density $(10^{-6}$ erg/s/cm$^2$/$\AA$)')    
    plt.savefig(saveDir+saveName+'_Extp.png')
    plt.close(fig)
    print('### Plotted in', saveDir+saveName+'_Extp.png')

    ## Save the result
    if all(np.isfinite(fitParameter)):
        output = pd.concat([model.resample[np.isfinite(model.resample['binned_flux'])], 
                            extrapolation], 
                           axis=0, join='inner')
        ## Convert the unit of flux from 1e-6 erg/s/cm^2/AA to nJy
        outputFlux = np.array(output['binned_flux']*((output['wavelength']*10.)**2)/\
                              (constants.speed_of_light*1e10)*1e38)
        fitsCol = fits.Column(name='Flux', array=outputFlux, format='E', unit='nJy')
        fitsHeader = fits.Header()
        fitsHeader['CRPIX1'] = 1
        fitsHeader['CRVAL1'] = output.iat[0,0]
        fitsHeader['CDELT1'] = boxSmoothWidth
        fitsHeader['CUNIT1'] = 'nm'
        fitsHeader['CTYPE1'] = 'Wavelength'

        fitsHeader['EXTF1'] = (fitParameter[0], 'Fitting parameter a for extrapolation')
        fitsHeader['EXTF2'] = (fitParameter[1], 'Fitting parameter b for extrapolation')
        fitsHeader['EXTF3'] = (fitParameter[2], 'Fitting parameter c for extrapolation')
        fitsHeader['COMMENT'] = 'Fitting function: flux[erg/s/cm^2/A]=a*exp(-b*wavelength[nm])+c'
        fitsHeader['COMMENT'] = 'Original AMBRE data: ' + baseName

        AmbreParam = baseName.split(':')
        fitsHeader['ATMMODEL'] = (baseName[:1], 'Plan-parallel or Spherical geometry')
        fitsHeader['TEFF'] =  (baseName[1:5] , 'Teff (K)')
        fitsHeader['GRAVITY'] = (AmbreParam[1][1:5] , 'Log of surface gravity (cm/s^2)')
        fitsHeader['MASS'] = (AmbreParam[2][1:4] , 'Mass of the MARCS model atmosphere (Msun)')
        fitsHeader['TURBV'] = (AmbreParam[3][1:3] , 'Microturbulence velocity (km/s)') 
        fitsHeader['METAL'] = (AmbreParam[4][1:7] , 'Mean stellar metallicity ([Metal/H or [Fe/H])')
        fitsHeader['ALPHAE'] = (AmbreParam[5][1:6] , '[alpha/Fe] chemical index')

        fitsHeader['HISTORY'] = ('Produced from ' + os.path.basename(__file__) + 
                                ' on ' + str(datetime.date.today()))

        outputFits = fits.BinTableHDU.from_columns([fitsCol], fitsHeader)
        outputFits.writeto(saveDir+saveName+'_Extp.fits', overwrite=True)
        print('### Result was saved in', saveDir+saveName+'_Extp.fits')

    ## Add a summry to the summary file
    summaryData = pd.DataFrame({'ModelName': saveName,
                                'a': [fitParameter[0]], 'a_e': [fitParameterError[0]],
                                'b': [fitParameter[1]], 'b_e': [fitParameterError[1]],
                                'c': [fitParameter[2]], 'c_e': [fitParameterError[2]],
                                'std': [fitSTD],
                                'r': [correlationCoefficien]
                                })
    summaryFileName = saveDir+'summary.dat'
    if i == 0:
        if os.path.isfile(summaryFileName):
            os.remove(summaryFileName)
        summaryData.to_csv(summaryFileName, index=False, sep=',', mode='w', na_rep='NaN')
    else:
        summaryData.to_csv(summaryFileName, index=False, sep=',', mode='a', header=False, na_rep='NaN')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('listFile', help='Name of a list file descripting paths to spectra.')
    parser.add_argument('saveDir', help='Path to a directory to save generated files.')
    parser.add_argument('--fitRange', nargs='*', type=float, default=[980.0, 1180.0], 
                        help='Fitting range in nm')
    parser.add_argument('--boxSmoothWidth', type=float, default=0.01, help='Width of a boxcar kernel in nm')
    parser.add_argument('--gaussSmoothWidth', type=float, default=0.24, 
                        help='FWHM of Gaussian to smooth a spectrum in nm')
    parser.add_argument('--paschenWidth', type=float, default=30.0, help='FWHM of Paschen absorptions in nm')
    args = parser.parse_args()

    ## Loop
    with open(args.listFile) as listTable:
        fileList = [line.rstrip() for line in listTable]
        if os.path.isfile(args.saveDir+'summary.dat'):
            processLog = pd.read_csv(args.saveDir+'summary.dat')
            processedModel = np.array(processLog['ModelName'])
            for i, dataPath in enumerate(fileList):
                if os.path.basename(dataPath) in processedModel:
                    print('### Process has been already done.', i, dataPath)
                else:
                    print('### Start processing:', i, dataPath)
                    doProcessing(dataPath, i, args.saveDir, args.fitRange, args.boxSmoothWidth, 
                                 args.gaussSmoothWidth, args.paschenWidth)
                    print('### Finished.')
                    print('   ')
        else:
            for i, dataPath in enumerate(fileList):
                print('### Start processing:', i, dataPath)
                doProcessing(dataPath, i, args.saveDir, args.fitRange, args.boxSmoothWidth, 
                             args.gaussSmoothWidth, args.paschenWidth)
                print('### Finished.')
             

if __name__ == '__main__':
    main()

