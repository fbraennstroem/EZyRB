"""Module for the Reduced Order Modeling."""

import math
import copy
import pickle
import numpy as np
from scipy.spatial.qhull import Delaunay
from sklearn.model_selection import KFold
from .database import Database

# ==============================================================================
# UoW adjustment
def createDir(outputDir):
    """
    TODO: only for old included post-processing; needs removal
    Function to make directory if not available
    """
    if not(os.path.isdir(outputDir)):
        try:
            os.makedirs(outputDir)
        except OSError as e:
            if e.errno != 17:
                raise
            pass
# End UoW
# ==============================================================================

class ReducedOrderModel():
    """
    Reduced Order Model class.

    This class performs the actual reduced order model using the selected
    methods for approximation and reduction.

    :param ezyrb.Database database: the database to use for training the
        reduced order model.
    :param ezyrb.Reduction reduction: the reduction method to use in reduced
        order model.
    :param ezyrb.Approximation approximation: the approximation method to use
        in reduced order model.
    :param object scaler_red: the scaler for the reduced variables (eg. modal
        coefficients). Default is None.

    :cvar ezyrb.Database database: the database used for training the reduced
        order model.
    :cvar ezyrb.Reduction reduction: the reduction method used in reduced order
        model.
    :cvar ezyrb.Approximation approximation: the approximation method used in
        reduced order model.
    :cvar object scaler_red: the scaler for the reduced variables (eg. modal
        coefficients).

    :Example:

         >>> from ezyrb import ReducedOrderModel as ROM
         >>> from ezyrb import POD, RBF, Database
         >>> pod = POD()
         >>> rbf = RBF()
         >>> # param, snapshots and new_param are assumed to be declared
         >>> db = Database(param, snapshots)
         >>> rom = ROM(db, pod, rbf).fit()
         >>> rom.predict(new_param)

    """
    def __init__(self, database, reduction, approximation,
                 plugins=None):

        self.database = database
        self.reduction = reduction
        self.approximation = approximation

        if plugins is None:
            plugins = []

        self.plugins = plugins

    # ==============================================================================
    # Start UoW - adjustments

    def test_error_norm(self, predicted_snapshot, test_snapshot, norm=np.linalg.norm):
        """
        Compute error norm for given single snapshot
        """
        return np.mean(
            norm(predicted_snapshot - test_snapshot, axis=1) /
            norm(test_snapshot, axis=1))

    def test_error_local_db(self, test, norm=np.linalg.norm):
        """
        Compute the mean norm of the relative error vectors of predicted
        test snapshots.

        """
        predicted_test = self.predict(test.parameters)

        return np.mean(
            norm(predicted_test - test.snapshots, axis=1) /
            norm(test.snapshots, axis=1))

    def errorUoW(self,new_db,idL, fieldName, meshData, n_splits, shuffleKFold, csvOutput, plotDiagrams, plotDiagramStep, debugLevel, outputFile, *args, norm=np.linalg.norm, **kwargs):
        """
        TODO: check if needed - Approximation / Reduction and Fitting of external kfold data
        """
        print(new_db)
        rom = type(self)(new_db, copy.deepcopy(self.reduction), copy.deepcopy(self.approximation)).fit(*args, **kwargs)
        print(self.database[test_index])
        errorPerFold = rom.test_error(self.database[test_index], norm)
        return np.array(errorPerFold)

    def reduce_approximate_rom(self,trainDB, fieldName, meshData, n_splits, *args, norm=np.linalg.norm, **kwargs):
        """
        Already splitted dabase is reduced and approximated; kfolding is done separately in main script.
        """
        rom = type(self)(trainDB, copy.deepcopy(self.reduction),
                         copy.deepcopy(self.approximation)).fit(
                             *args, **kwargs)
        # return rom

    # def kfold_cv_errorUoW(self,idL, fieldName, meshData, n_splits, shuffleKFold, csvOutput, plotDiagrams, plotDiagramStep, debugLevel, outputFile, *args, norm=np.linalg.norm, **kwargs):
    #     r"""
    #     Includes also post-processing - but is actually not needed anymore - old non-flexible approach
    #     """
    #     error = []
    #     if shuffleKFold:
    #         kf = KFold(n_splits=n_splits,shuffle=True)
    #     else:
    #         kf = KFold(n_splits=n_splits,shuffle=False)
    #     kfoldNr = 0
    #     print (100*"-")

    #     # ==============================================================================
    #     for train_index, test_index in kf.split(self.database):
    #         new_db = self.database[train_index]
    #         parameterV = self.database[train_index].parameters
    #         snapshotsV = self.database[train_index].snapshots
    #         # print(parameterV)
    #         # print(snapshotsV)
    #         parameterTestV = self.database[test_index].parameters
    #         print (100*"U")
    #         print(parameterTestV)
    #         # exit()
    #         # new_db = Database(parameterV, snapshots)
    #         # print(new_db)
    #         # new_db= (new_db[:,1:])
    #         # print(new_db)
    #         # TODO remove first column with IDs
    #         # print(self.reduction)
    #         # print(dir(self.reduction))
    #         # print(self.approximation)
    #         # print(dir(self.approximation))
    #         rom = type(self)(new_db, copy.deepcopy(self.reduction), copy.deepcopy(self.approximation)).fit(*args, **kwargs)
    #         errorPerFold = rom.test_error(self.database[test_index], norm)
    #         error.append(errorPerFold)
    #         # print("test.parameters: ",self.database[test_index].parameters)

    #         testData = self.database[test_index]
    #         # print(testData.parameters)
    #         # testData = (testData.parameters[:,1:])
    #         # idNumberL = (testData.parameters[:,0])
    #         # # print(idNumberL)
    #         # exit()
    #         # TODO remove first column with IDs

    #         # print(testData.parameters)
    #         # print(testData.snapshots)
    #         # exit()
    #         print (100*".")
    #         print("kfoldNr: ",kfoldNr)
    #         numberTrainingData = len(testData.parameters) * n_splits
    #         print(" - total number of trainings data: ",numberTrainingData)
    #         print(" - number of data per fold: ",len(testData.parameters))
    #         print(" - avg. error per fold: ", np.round(errorPerFold,5))
    #         # print((testData.parameters))
    #         # print((testData.parameters.shape))

    #         if plotDiagrams:
    #             outDir1 = "./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData) +"__slices/"
    #             createDir(outDir1)
    #             outDir1a = "./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData) +"__slices/ROM/"
    #             createDir(outDir1a)
    #             outDir1b = "./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData) +"__slices/FOM/"
    #             createDir(outDir1b)
    #             outDir1c = "./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData) +"__slices/rel_DIFF/"
    #             createDir(outDir1c)
    #             outDir1d = "./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData) +"__slices/abs_DIFF/"
    #             createDir(outDir1d)
    #             outDir1e = "./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData) +"__slices/FOM_ROM/"
    #             createDir(outDir1e)
    #             # outDir1f = "./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData) +"__slices/rel_DIFF_range/"
    #             # createDir(outDir1f)

    #             outDir3 ="./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData)+"__histogram/"
    #             createDir(outDir3)
    #             # outDir4 ="./POST/"+fieldName +"/trainDataSize_idn_error_param_" +str(numberTrainingData)+"__csv/"
    #             # createDir(outDir4)
    #             # outputfile.write("fieldName,idn, numberTraining, error,param1, param2, param3, param4\n")

    #         if csvOutput:
    #             outDir2 ="./POST/"+fieldName +"/trainDataSize_" +str(numberTrainingData)+"__csv/"
    #             createDir(outDir2)

    #         if fieldName == "TEMPERATURE":
    #             unit = " (C)"
    #         elif fieldName == "W-VELOCITY":
    #             unit = " (m/s)"
    #         elif fieldName == "HRRPUV":
    #             unit = " (kW/m3)"
    #         else:
    #             unit = " (-)"

    #         for n, param in enumerate(testData.parameters):
    #             if debugLevel == 1:
    #                 # print ("idNumber: ", idShuffeldL[n])
    #                 print (100*"x")
    #                 print ("n (parameter number): ", n)
    #                 print("Parameter for testing: ", param)
    #             # exit()
    #             predicted_test = rom.predict(param)
    #             # print(predicted_test)
    #             # print (100*"-")
    #             # print ("Prediction: ", predicted_test.shape)
    #             # print ("snapshot n: ", testData.snapshots.shape)
    #             # print ("snapshot n: ", testData.snapshots[n].shape)
    #             abs_diff = predicted_test - testData.snapshots[n]
    #             # print("abs_diff: ", abs_diff.shape)
    #             if min(testData.snapshots[n]) == 0.0:
    #                 rel_diff = (predicted_test - testData.snapshots[n]) / (testData.snapshots[n] + 0.00000000001)
    #             else:
    #                 rel_diff = (predicted_test - testData.snapshots[n]) / testData.snapshots[n]
    #             idn = str(random.randint(1, 100000)) 

    #             # print(norm(predicted_test - testData.snapshots, axis=1))
    #             # print(norm(testData.snapshots, axis=1))
    #             # print(norm(predicted_test - testData.snapshots, axis=1).shape)
    #             # print(norm(testData.snapshots, axis=1).shape)
    #             # norm_diff = norm(predicted_test - testData.snapshots, axis=1) / norm(testData.snapshots, axis=1)
    #             norm_diff = norm(predicted_test - testData.snapshots, axis=1) / norm(testData.snapshots, axis=1)
    #             # print("Normdiff: ", norm_diff.shape)

    #             norm_diff_mean = np.mean(norm(predicted_test - testData.snapshots, axis=1) / norm(testData.snapshots, axis=1))

    #             # ==============================================================================
    #             # CSV
    #             if csvOutput:
    #                 np.savetxt(outDir2 + fieldName + "_" + idn +"__nrTest_"+str(numberTrainingData) +"_"+str(kfoldNr) +"_nSplits_"+str(n_splits) +"_"+ '_abs_DIFF.csv', abs_diff, delimiter=',')
    #                 np.savetxt(outDir2 + fieldName + "_" + idn +"__nrTest_"+str(numberTrainingData) +"_"+str(kfoldNr)+"_nSplits_"+str(n_splits) +"__"+ '_rel_DIFF.csv', rel_diff, delimiter=',')

    #             if plotDiagrams:
    #                 if n % plotDiagramStep == 0:
    #                     # ==============================================================================
    #                     MeshSize_x = 20
    #                     MeshSize_y = 45

    #                     # ..............................................................................
    #                     matrix_rom = predicted_test
    #                     matrix_rom = matrix_rom.reshape((MeshSize_x+1),(MeshSize_y+1))
    #                     matrix_rom = np.rot90(matrix_rom, 1)

    #                     plt.figure(figsize=(4.2,6))
    #                     plt.imshow(matrix_rom, cmap='viridis')#, vmin=, vmax=max_Range_of_colorbar)
    #                     plt.colorbar(label=fieldName+unit)
    #                     filename = outDir1a + fieldName+"_"+idn +"__nrTest_"+str(numberTrainingData) +"__kfold_"+str(kfoldNr) +"__nSplits_"+str(n_splits) + '__ROM.png'
    #                     plt.xlabel("x-Position (m)")
    #                     plt.ylabel("z-Position (m)")
    #                     plt.title(param)
    #                     plt.savefig(filename)
    #                     plt.close()

    #                     # ..............................................................................
    #                     matrix = testData.snapshots[n]
    #                     matrix = matrix.reshape((MeshSize_x+1),(MeshSize_y+1))
    #                     matrix = np.rot90(matrix, 1)
    #                     # print(matrix.max())
    #                     # print(matrix.min())

    #                     plt.figure(figsize=(4.2,6))
    #                     plt.imshow(matrix, cmap='viridis')#, vmin=, vmax=max_Range_of_colorbar)
    #                     plt.colorbar(label=fieldName+unit)
    #                     filename = outDir1b + fieldName +"_"+idn+"__nrTest_"+str(numberTrainingData) +"__kfold_"+str(kfoldNr) +"__nSplits_"+str(n_splits) + '__FOM.png'
    #                     plt.xlabel("x-Position (m)")
    #                     plt.ylabel("z-Position (m)")
    #                     plt.title(param)
    #                     plt.savefig(filename)
    #                     plt.close()

    #                     # ..............................................................................
    #                     # fig = plt.figure(layout='constrained', figsize=(10, 4))
    #                     # subfigs = fig.subfigures(1, 2, wspace=0.07)
    #                     # plt.figure(figsize=(4.2,6))
    #                     # fig, (ax1, ax2) = plt.subplots(1, 2)
    #                     fig, axs = plt.subplots(1, 2)
    #                     # fig.suptitle(param)
    #                     fig.suptitle("L2-norm: "+ str(np.round(norm_diff_mean,5)))#+" \n Parameters: " + str(param))
    #                     # axs[1] = plt.imshow(matrix_rom, cmap='viridis')#, vmin=, vmax=max_Range_of_colorbar)
    #                     # axs[0] = plt.imshow(matrix, cmap='viridis')#, vmin=, vmax=max_Range_of_colorbar)
    #                     axs[0].set_title('FOM')
    #                     if debugLevel == 2:
    #                         print("-- debugLevel 2 --------------------------------")
    #                         print ("  - FOM max range: ", matrix.max()) 
    #                         print ("  - FOM min range: ", matrix.min()) 
    #                         print ("  - ROM max range: ", matrix_rom.max()) 
    #                         print ("  - ROM min range: ", matrix_rom.min()) 
    #                     # max_Range_of_colorbar = 1200.0
    #                     max_Range_of_colorbar = matrix.max()
    #                     min_Range_of_colorbar = matrix.min()
    #                     im1= axs[0].imshow(matrix, cmap='viridis', vmin=min_Range_of_colorbar, vmax=max_Range_of_colorbar)
    #                     axs[1].set_title('ROM')
    #                     im2 = axs[1].imshow(matrix_rom, cmap='viridis', vmin=min_Range_of_colorbar, vmax=max_Range_of_colorbar)
    #                     # plt.colorbar(label=fieldName+unit)
    #                      # cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(0.2), format="%.2f")
    #                     fig.colorbar(im2, orientation='vertical', label=fieldName+unit)

    #                     filename = outDir1e + fieldName +"_"+idn+"__nrTest_"+str(numberTrainingData) +"__kfold_"+str(kfoldNr) +"__nSplits_"+str(n_splits) + '__FOM_ROM.png'
    #                     plt.xlabel("x-Position (m)")
    #                     plt.ylabel("z-Position (m)")
    #                     # plt.title(param)
    #                     plt.savefig(filename)
    #                     plt.close()

    #                     # # ..............................................................................
    #                     matrix = rel_diff
    #                     matrix = matrix.reshape((MeshSize_x+1),(MeshSize_y+1))
    #                     matrix = np.rot90(matrix, 1)

    #                     # plt.figure(figsize=(4.2,6))
    #                     fig, axs = plt.subplots(1, 2)
    #                     # fig.suptitle(param)
    #                     fig.suptitle("L2-norm: "+ str(np.round(norm_diff_mean,5)))#+" \n Parameters: " + str(param))
    #                     axs[0].set_title('no range')
    #                     axs[1].set_title('-0.15 to 0.15')
    #                     im1 = axs[0].imshow(matrix, cmap='viridis')#, vmin=, vmax=max_Range_of_colorbar)
    #                     im2 = axs[1].imshow(matrix, cmap='viridis', vmin=-0.15, vmax=0.15)
    #                     # plt.colorbar(label=fieldName+unit)
    #                     fig.colorbar(im1, orientation='vertical')
    #                     fig.colorbar(im2, orientation='vertical', label=fieldName+unit)
    #                     filename = outDir1c + fieldName +  "_"+ idn + "_l2_"+ str(np.round(norm_diff_mean,4)) + "__nrTest_"+str(numberTrainingData) +"__kfold_"+str(kfoldNr) +"__nSplits_"+str(n_splits) + '_rel_DIFF.png'
    #                     # plt.title("L2-norm: "+ str(np.round(norm_diff_mean,5)))
    #                     # plt.xlabel("x-Position (m)")
    #                     # plt.ylabel("z-Position (m)")
    #                     # plt.title(param)
    #                     plt.savefig(filename)
    #                     plt.close()

    #                     # ..............................................................................
    #                     # matrix = norm_diff
    #                     # matrix = matrix.reshape((MeshSize_x+1),(MeshSize_y+1))
    #                     # matrix = np.rot90(matrix, 1)
    #                     # plt.imshow(matrix, cmap='viridis')#, vmin=, vmax=max_Range_of_colorbar)
    #                     # plt.colorbar(label='Temperature (C)')
    #                     # filename = fieldName +"_"+str(kfoldNr) +"_nSplits_"+str(n_splits) +"_"+ idn + '_normDiff_diff_HIST_pred_sol__DIFF_normDiff_SLICE.png'
    #                     # plt.title("L2-norm: ", norm_diff_mean)
    #                     # plt.savefig(filename)
    #                     # plt.close()

    #                     # ..............................................................................
    #                     matrix = abs_diff
    #                     matrix = matrix.reshape((MeshSize_x+1),(MeshSize_y+1))
    #                     matrix = np.rot90(matrix, 1)

    #                     plt.figure(figsize=(4.2,6))
    #                     plt.imshow(matrix, cmap='viridis')#, vmin=, vmax=max_Range_of_colorbar)
    #                     plt.colorbar(label=fieldName+unit)
    #                     plt.xlabel("x-Position (m)")
    #                     plt.ylabel("z-Position (m)")
    #                     plt.title("L2-norm: "+ str(np.round(norm_diff_mean,5))+" "+str(param))
    #                     filename = outDir1d + fieldName +"_"+idn+ "_l2_"+ str(np.round(norm_diff_mean,4)) +"__nrTest_"+str(numberTrainingData) +"__kfold_"+str(kfoldNr) +"__nSplits_"+str(n_splits) + '_abs_DIFF.png'
    #                     plt.savefig(filename)
    #                     plt.close()


    #                     # ==============================================================================
    #                     n_bins = 100
    #                     plt.hist(rel_diff, n_bins, histtype='step', stacked=True, fill=False)
    #                     plt.xlabel("L2-norm")
    #                     plt.xlabel("Relative Error (%)")
    #                     # plt.ylabel("(m)")
    #                     plt.title(param)
    #                     plt.savefig(outDir3 + fieldName +"_"+ idn +"__nrTest_"+str(numberTrainingData) +"__kfold_"+str(kfoldNr) +"_nSplits_"+str(n_splits) +"__rel_DIFF.png")
    #                     plt.close()

    #                     # plt.hist(rel_diff, n_bins, histtype='step', stacked=True, fill=False)
    #                     # plt.xlabel("L2-norm")
    #                     # plt.xlabel("x-Position (m)")
    #                     # plt.ylabel("z-Position (m)")
    #                     # plt.savefig(outDir3 + fieldName +"__nrTrain_"+str(numberTrainingData) +"__kfold_"+str(kfoldNr) +"_nSplits_"+str(n_splits) +"_"+ idn + '_rel_DIFF.png')
    #                     # outputFile.write("%s, %s, %d, %s, %.4f,  %.4f,  %.4f,  %.4f\n" % (fieldName,idn, numberTrainingData, str(np.round(norm_diff_mean,5)) ,param[0], param[1], param[2], param[3]))
    #                     paramStrL = []
    #                     for par in param:
    #                         # print(par)
    #                         paramStrL.append(str(par))
    #                     paramStr = ",".join(paramStrL)
    #                     # print(paramStr)
    #                     # outputFile.write("%s, %s, %d, %s, %s\n" % (fieldName,idn, numberTrainingData, str(np.round(norm_diff_mean,5)) ,paramStrL))
    #                     outputFile.write("%s,%s,%d,%s,%s\n" % (fieldName,idn, numberTrainingData, str(np.round(norm_diff_mean,5)) ,paramStr))
    #                     # exit()
    #                     # outputFile.write("%s, %s, %d, %s, %.4f,  %.4f,  %.4f,  %.4f\n" % (fieldName,idn, numberTrainingData, str(np.round(norm_diff_mean,5)) ,param[0], param[1], param[2], param[3]))
    #                     # plt.close()
    #                     # exit()
    #         kfoldNr = kfoldNr + 1

    #         # for each_set in test_data
    #         # error = [1.0,2.0]

    #     return np.array(error)


    # End UoW
    # ==============================================================================

    def fit(self):
        r"""
        Calculate reduced space

        """

        import copy
        self._full_database = copy.deepcopy(self.database)

        # FULL-ORDER PREPROCESSING here
        for plugin in self.plugins:
            plugin.fom_preprocessing(self)

        self.reduction.fit(self._full_database.snapshots_matrix.T)
        # print(self.reduction.singular_values)
        # print(self._full_database.snapshots_matrix)
        reduced_snapshots = self.reduction.transform(
            self._full_database.snapshots_matrix.T).T

        self._reduced_database = Database(
            self._full_database.parameters_matrix, reduced_snapshots)

        # REDUCED-ORDER PREPROCESSING here
        for plugin in self.plugins:
            plugin.rom_preprocessing(self)

        self.approximation.fit(
            self._reduced_database.parameters_matrix,
            self._reduced_database.snapshots_matrix)

        return self

    def predict(self, mu):
        """
        Calculate predicted solution for given mu

        :return: the database containing all the predicted solution (with
            corresponding parameters).
        :rtype: Database
        """
        print("mu: ",mu)
        print("self.database: ",self.database)
        # print("self.database: ",self.database.parameters)
        mu = np.atleast_2d(mu)

        self._reduced_database = Database(
                mu, np.atleast_2d(self.approximation.predict(mu)))

        # REDUCED-ORDER POSTPROCESSING here
        for plugin in self.plugins:
            plugin.rom_postprocessing(self)

        self._full_database = Database(
            np.atleast_2d(mu),
            self.reduction.inverse_transform(
                    self._reduced_database.snapshots_matrix.T).T
        )

        # FULL-ORDER POSTPROCESSING here
        for plugin in self.plugins:
            plugin.fom_postprocessing(self)

        return self._full_database

    def save(self, fname, save_db=True, save_reduction=True, save_approx=True):
        """
        Save the object to `fname` using the pickle module.

        :param str fname: the name of file where the reduced order model will
            be saved.
        :param bool save_db: Flag to select if the `Database` will be saved.
        :param bool save_reduction: Flag to select if the `Reduction` will be
            saved.
        :param bool save_approx: Flag to select if the `Approximation` will be
            saved.

        Example:

        >>> from ezyrb import ReducedOrderModel as ROM
        >>> rom = ROM(...) #  Construct here the rom
        >>> rom.fit()
        >>> rom.save('ezyrb.rom')
        """
        rom_to_store = copy.copy(self)

        if not save_db:
            del rom_to_store.database
        if not save_reduction:
            del rom_to_store.reduction
        if not save_approx:
            del rom_to_store.approximation

        with open(fname, 'wb') as output:
            pickle.dump(rom_to_store, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        """
        Load the object from `fname` using the pickle module.

        :return: The `ReducedOrderModel` loaded

        Example:

        >>> from ezyrb import ReducedOrderModel as ROM
        >>> rom = ROM.load('ezyrb.rom')
        >>> rom.predict(new_param)
        """
        with open(fname, 'rb') as output:
            rom = pickle.load(output)

        return rom

    def test_error(self, test, norm=np.linalg.norm):
        """
        Compute the mean norm of the relative error vectors of predicted
        test snapshots.

        :param database.Database test: the input test database.
        :param function norm: the function used to assign at the vector of
            errors a float number. It has to take as input a 'numpy.ndarray'
            and returns a float. Default value is the L2 norm.
        :return: the mean L2 norm of the relative errors of the estimated
            test snapshots.
        :rtype: numpy.ndarray
        """
        predicted_test = self.predict(test.parameters_matrix)
        return np.mean(
            norm(predicted_test.snapshots_matrix - test.snapshots_matrix,
            axis=1) / norm(test.snapshots_matrix, axis=1))


    def kfold_cv_error(self, n_splits, *args, norm=np.linalg.norm, **kwargs):
        r"""
        Split the database into k consecutive folds (no shuffling by default).
        Each fold is used once as a validation while the k - 1 remaining folds
        form the training set. If `n_splits` is equal to the number of
        snapshots this function is the same as `loo_error` but the error here
        is relative and not absolute.

        :param int n_splits: number of folds. Must be at least 2.
        :param function norm: function to apply to compute the relative error
            between the true snapshot and the predicted one.
            Default value is the L2 norm.
        :param \*args: additional parameters to pass to the `fit` method.
        :param \**kwargs: additional parameters to pass to the `fit` method.
        :return: the vector containing the errors corresponding to each fold.
        :rtype: numpy.ndarray
        """
        error = []
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(self.database):
            new_db = self.database[train_index]
            rom = type(self)(new_db, copy.deepcopy(self.reduction),
                             copy.deepcopy(self.approximation)).fit(
                                 *args, **kwargs)

            error.append(rom.test_error(self.database[test_index], norm))

        return np.array(error)

    def loo_error(self, *args, norm=np.linalg.norm, **kwargs):
        r"""
        Estimate the approximation error using *leave-one-out* strategy. The
        main idea is to create several reduced spaces by combining all the
        snapshots except one. The error vector is computed as the difference
        between the removed snapshot and the projection onto the properly
        reduced space. The procedure repeats for each snapshot in the database.
        The `norm` is applied on each vector of error to obtained a float
        number.

        :param function norm: the function used to assign at each vector of
            error a float number. It has to take as input a 'numpy.ndarray` and
            returns a float. Default value is the L2 norm.
        :param \*args: additional parameters to pass to the `fit` method.
        :param \**kwargs: additional parameters to pass to the `fit` method.
        :return: the vector that contains the errors estimated for all
            parametric points.
        :rtype: numpy.ndarray
        """
        error = np.zeros(len(self.database))
        db_range = list(range(len(self.database)))

        for j in db_range:
            indeces = np.array([True] * len(self.database))
            indeces[j] = False

            new_db = self.database[indeces]
            test_db = self.database[~indeces]
            rom = type(self)(new_db, copy.deepcopy(self.reduction),
                             copy.deepcopy(self.approximation)).fit()

            error[j] = rom.test_error(test_db)

        return error

    def optimal_mu(self, error=None, k=1):
        """
        Return the parametric points where new high-fidelity solutions have to
        be computed in order to globally reduce the estimated error. These
        points are the barycentric center of the region (simplex) with higher
        error.

        :param numpy.ndarray error: the estimated error evaluated for each
            snapshot; if error array is not passed, it is computed using
            :func:`loo_error` with the default function. Default value is None.
        :param int k: the number of optimal points to return. Default value is
            1.
        :return: the optimal points
        :rtype: numpy.ndarray
        """
        if error is None:
            error = self.loo_error()

        mu = self.database.parameters_matrix
        tria = Delaunay(mu)

        error_on_simplex = np.array([
            np.sum(error[smpx]) * self._simplex_volume(mu[smpx])
            for smpx in tria.simplices
        ])

        barycentric_point = []
        for index in np.argpartition(error_on_simplex, -k)[-k:]:
            worst_tria_pts = mu[tria.simplices[index]]
            worst_tria_err = error[tria.simplices[index]]

            barycentric_point.append(
                np.average(worst_tria_pts, axis=0, weights=worst_tria_err))

        return np.asarray(barycentric_point)

    def _simplex_volume(self, vertices):
        """
        Method implementing the computation of the volume of a N dimensional
        simplex.
        Source from: `wikipedia.org/wiki/Simplex
        <https://en.wikipedia.org/wiki/Simplex>`_.

        :param numpy.ndarray simplex_vertices: Nx3 array containing the
            parameter values representing the vertices of a simplex. N is the
            dimensionality of the parameters.
        :return: N dimensional volume of the simplex.
        :rtype: float
        """
        distance = np.transpose([vertices[0] - vi for vi in vertices[1:]])
        return np.abs(
            np.linalg.det(distance) / math.factorial(vertices.shape[1]))
