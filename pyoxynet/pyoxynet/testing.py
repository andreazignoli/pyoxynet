import pandas as pd
import numpy as np
from pandas import read_csv

class Test:

    def __init__(self, filename):
        self.filename = filename

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_data_extension(self, data_extension):
        self.data_extension = data_extension

    def clear_file_name(self):
        if '.' in self.filename and self.filename.split('.')[-1]=='.csv':
            self.filename_cleared = self.filename.split('.')[0]
        else:
            self.filename_cleared = self.filename

        self.filename_cleared = self.filename_cleared.rpartition('/')[-1]

    def infer_metabolimeter(self):

        import chardet

        try:
            if self.data_extension == '.csv':
                df = read_csv(self.filename + self.data_extension)
                print('Just reading a csv file')
            if self.data_extension == '.txt':
                df = read_csv(self.filename + self.data_extension, sep="\t", header=None, skiprows=3)
                print('Just reading a txt file')
                self.metabolimeter = 'vyiare'
        except:
            f = open(self.filename + self.data_extension, encoding="utf8", errors="ignore")
            df = read_csv(f, header=0)

        self.df = df

        if 'VO2_I' in df.columns:
            self.metabolimeter = 'generated_pyoxynet'

        if 'Polso O2' in df.columns:
            self.metabolimeter = 'vintus'

        if df.isnull().apply(lambda x: all(x), axis=1)[0]:
            # print('Most probably we have a Cortex file at first sight')
            # print(df.columns[0])
            # print(df)
            self.metabolimeter = 'cortex'

        try:
            if df['Unnamed: 0'][0] == 'CPET Results':
                # print('Most probably we have a Cortex file at second sight')
                # print(df)
                self.metabolimeter = 'cortex'
        except:
            pass

        try:
            if 'Messzeit' in df['ID:'].values:
                # print('Most probably we have a Cortex file at second sight')
                # print(df)
                self.metabolimeter = 'cortex_bruce'
        except:
            pass

        try:
            if 'Patient' in df.columns[0]:
                # print('Most probably we have a Cortex file at second sight')
                # print(df)
                self.metabolimeter = 'cortex_bruce_2'
        except:
            pass

        try:
            if 'Marqueur' in df[df.columns[2]].tolist():
                print('Most probably we have a Cortex file that Bruce provided')
                self.metabolimeter = 'cortex_bruce_3'

            if 'Work' in df.columns[2]:
                print('Most probably we have an Italian file from cardiac patients (CENTRO MONZINO)')
                self.metabolimeter = 'centro-monzino'

            if 'MINUTE' in df.columns[0] or 'NOM :' in df.columns[0]:
                print('Most probably we have a Mourot file from cardiac patients')
                self.metabolimeter = 'mourot_cardiac'

            if 'Time' in df.columns[0] and 'Load' in df.columns[1]:
                print('Most probably we have a Low file')
                self.metabolimeter = 'low'

            if 'Time' in df.columns[0] and 'Time' in df.columns[1]:
                print('Most probably we have a VO2Master file')
                self.metabolimeter = 'VO2Master'

            if 'Zeit' in df.columns[0]:
                print('Most probably we have a Unisbz file')
                self.metabolimeter = 'unisbz'

            if 'Codice' in df.columns[0]:
                print('Most probably we have a Italian Cosmed file')
                self.metabolimeter = 'cosmed'

            if df[df.columns[0]][0] == 'Nom de famille':
                print('Most probably we have a French Cosmed file')
                self.metabolimeter = 'cosmed'

            if df[df.columns[0]][0] == 'Sobrenome':
                print('Most probably we have a Brasilian Cosmed file')
                self.metabolimeter = 'cosmed'

            if 'ID code' in df.columns[0]:
                print('Most probably we have an English Cosmed file')
                self.metabolimeter = 'cosmed'

            if 'Code ID' in df.columns[0]:
                print('Most probably we have an English Cosmed file')
                self.metabolimeter = 'cosmed'

            if 'ID1' in df.columns[0]:
                print('Most probably we have an English Cosmed file')
                self.metabolimeter = 'cosmed'

            if 'Temps' in df[df.columns[0]].unique():
                print('Most probably we have a French Mourot file')
                self.metabolimeter = 'mourot'

            if 'TEMPS' in df.columns[0]:
                print('Most probably we have an French file')
                self.metabolimeter = 'mourot_COPD'

        except:
            pass

    def load_file(self):
        from . import utilities

        df = self.df

        if self.metabolimeter == 'vintus':

            starting_index = df.loc[df['Tempo'] == 'min'].index
            ref_string = 'Tempo'
            ventilatory_data = df[starting_index[0]+1:-1]

            # print('Weight')
            self.weight = 70
            # print('Height')
            self.height = 180
            # print('Age')
            self.age = 40
            # print('Gender')
            self.gender = 'M'

            # print('Reading data')
            n_rows = len(ventilatory_data.index)
            # initialise variables
            self.time = np.zeros((n_rows,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows,), dtype=np.float32)
            self.HR = np.zeros((n_rows,), dtype=np.float32)
            self.Rf = np.zeros((n_rows,), dtype=np.float32)
            self.VE = np.zeros((n_rows,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows,), dtype=np.float32)
            self.load = np.zeros((n_rows,), dtype=np.float32)

            i = 0
            for time_sec in ventilatory_data[ventilatory_data.columns[0]].values:
                # print(i)
                try:
                    self.time[i] = utilities.get_sec(time_sec)
                    self.VO2[i] = float(ventilatory_data[ventilatory_data.columns[5]].values[i])
                    self.VCO2[i] = float(ventilatory_data[ventilatory_data.columns[6]].values[i])
                    self.HR[i] = float(ventilatory_data[ventilatory_data.columns[2]].values[i])

                    try:
                        self.load[i] = float(ventilatory_data[ventilatory_data.columns[1]].values[i])
                    except:
                        self.load[i] = 0

                    self.VE[i] = float(ventilatory_data[ventilatory_data.columns[3]].values[i])
                    self.Rf[i] = 0
                    self.PetO2[i] = float(ventilatory_data[ventilatory_data.columns[15]].values[i]) * 7.50062
                    self.PetCO2[i] = float(ventilatory_data[ventilatory_data.columns[14]].values[i]) * 7.50062
                    i += 1
                except:
                    pass

        if self.metabolimeter == 'vyiare':

            with open(self.filename + self.data_extension) as f:
                line1, line2 = next(f), next(f)

            # print('Weight')
            try:
                self.weight = int(line2[line2.find('Weight')+8:line2.find('Weight')+10])
                # print('Height')
                self.height = int(line1[line1.find('Height')+8:line1.find('Height')+11])
                # print('Age')
                self.age = 40
                # print('Gender')
                self.gender = line1[line1.find('Gender')+8].capitalize()
            except:
                self.weight = 70
                # print('Height')
                self.height = 170
                # print('Age')
                self.age = 40
                # print('Gender')
                self.gender = 'M'

            # print('Reading data')
            n_rows = len(df.index)
            # initialise variables
            self.time = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 3,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 3,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.load = np.zeros((n_rows - 3,), dtype=np.float32)

            i = 0
            for time_sec in df[0].values[2:-1]:
                # print(i)
                try:
                    self.time[i] = utilities.get_sec(time_sec)
                    self.VO2[i] = float(df[df.columns[4]].values[i+2])
                    self.VCO2[i] = float(df[df.columns[5]].values[i + 2])
                    self.HR[i] = 0
                    self.load[i] = 0
                    self.VE[i] = float(df[df.columns[6]].values[i + 2])
                    self.Rf[i] = float(df[df.columns[1]].values[i + 2])
                    self.PetO2[i] = float(df[df.columns[16]].values[i + 2])
                    self.PetCO2[i] = float(df[df.columns[17]].values[i + 2])
                    i += 1
                except:
                    pass

        if self.metabolimeter == 'cortex_bruce_3':

            f = open(self.filename + self.data_extension, encoding="utf8", errors="ignore")
            df = read_csv(f, header=1)

            try:
                starting_index = df.loc[df['Unnamed: 0'] == 't'].index
                ref_string = 'Unnamed: 0'
            except:
                starting_index = df.loc[df['Rsultats de TCP'] == 't'].index
                ref_string = 'Rsultats de TCP'

            ventilatory_data = df[starting_index[0]:-1]
            # rename columns
            ventilatory_data.columns = ventilatory_data.iloc[0]

            # print('Weight')
            self.weight = float(df[df[ref_string] == 'Poids'].values[0, 2].split(' ')[0].replace(',', '.'))
            # print('Height')
            self.height = float(df[df[ref_string] == 'Taille'].values[0, 2].split(' ')[0])
            # print('Age')
            self.age = 2020 - float(df[df[ref_string] == 'Date de Naissance'].values[0, 2].split('/')[2])
            # print('Gender')
            if df[df[ref_string] == 'Sexe'].values[0, 2].split(' ')[0] == 'femme':
                self.gender = 'F'
            else:
                self.gender = 'M'

            # print('Cortex with VT data inside the same file -> no need to have labels')
            VT_index = df.loc[df[ref_string] == 'Variable'].index
            threshold_data = df[VT_index[0]:VT_index[0] + 21]

            use_WR = False
            use_HR = False

            try:
                WRVT1 = float(threshold_data[threshold_data[ref_string] == 'TT'].values[0][5])
                try:
                    WRVT2 = float(threshold_data[threshold_data[ref_string] == 'TT'].values[0][8])
                except:
                    WRVT2 = 1000
                use_WR = True
            except:
                HRVT1 = float(threshold_data[threshold_data[ref_string] == 'FC'].values[0][5])
                try:
                    HRVT2 = float(threshold_data[threshold_data[ref_string] == 'FC'].values[0][8])
                except:
                    HRVT2 = 1000
                use_HR = True

            # print('Reading data')
            n_rows = len(ventilatory_data.index)
            # initialise variables
            self.time = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 3,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 3,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.load = np.zeros((n_rows - 3,), dtype=np.float32)

            i = 0
            for time_sec in ventilatory_data.t.values[2:-1]:
                # print(i)
                try:
                    self.time[i] = utilities.get_sec(time_sec)
                    self.VO2[i] = float(ventilatory_data[ventilatory_data.columns[3]].values[i + 2]) * 1000
                    self.VCO2[i] = float(ventilatory_data[ventilatory_data.columns[78]].values[i + 2]) * 1000
                    self.HR[i] = float(ventilatory_data[ventilatory_data.columns[6]].values[i + 2])

                    try:
                        self.load[i] = float(ventilatory_data[ventilatory_data.columns[7]].values[i + 2])
                    except:
                        self.load[i] = 0

                    self.VE[i] = float(ventilatory_data[ventilatory_data.columns[11]].values[i + 2])
                    self.Rf[i] = float(ventilatory_data[ventilatory_data.columns[13]].values[i + 2])
                    self.PetO2[i] = float(ventilatory_data[ventilatory_data.columns[62]].values[i + 2])
                    self.PetCO2[i] = float(ventilatory_data[ventilatory_data.columns[61]].values[i + 2])
                    i += 1
                except:
                    pass

            # self.time = self.time - self.time[0]
            if use_WR:
                VT1_index = np.where(self.load == int(WRVT1))[0][0]
                try:
                    VT2_index = np.where(self.load == int(WRVT2))[0][0]
                except:
                    VT2_index = -1
            elif use_HR:
                VT1_index = np.where(self.HR == int(HRVT1))[0][0]
                try:
                    VT2_index = np.where(self.HR == int(HRVT2))[0][0]
                except:
                    VT2_index = -1

            self.VT1 = self.time[VT1_index]
            self.VT2 = self.time[VT2_index]

        if self.metabolimeter == 'cortex':

            df = read_csv(self.filename + self.data_extension)

            try:
                starting_index = df.loc[df['Unnamed: 0'] == 't'].index
                ref_string = 'Unnamed: 0'
            except:
                try:
                    starting_index = df.loc[df['CPET Results'] == 't'].index
                    ref_string = 'CPET Results'
                except:
                    starting_index = df.loc[df['Ergebnisse des Spiroergometrietests'] == 't'].index
                    ref_string = 'Ergebnisse des Spiroergometrietests'

            ventilatory_data = df[starting_index[0]:-1]
            # rename columns
            ventilatory_data.columns = ventilatory_data.iloc[0]

            # print('Weight')
            try:
                self.weight = float(df[df[ref_string] == 'Weight'].values[0, 2].split(' ')[0])
            except:
                self.height = float(df[df[ref_string] == 'Gewicht'].values[0, 2].split(' ')[0].split(',')[0])
            # print('Height')
            try:
                self.height = float(df[df[ref_string] == 'Height'].values[0, 2].split(' ')[0])
            except:
                self.height = float(df[df[ref_string] == 'Größe'].values[0, 2].split(' ')[0])
            # print('Age')
            try:
                self.age = 2020 - float(df[df[ref_string] == 'Date of Birth'].values[0, 2].split('/')[2])
            except:
                self.age = 2020 - float(df[df[ref_string] == 'Geburtsdatum'].values[0, 2].split('.')[2])
            # print('Gender')
            try:
                if df[df[ref_string] == 'Sex'].values[0, 2].split(' ')[0] == 'male':
                    self.gender = 'M'
                else:
                    self.gender = 'F'
            except:
                self.gender = 'M'

            # print('Cortex with VT data inside the same file -> no need to have labels')
            try:
                VT_index = df.loc[df[ref_string] == 'Variable'].index
                threshold_data = df[VT_index[0]:VT_index[0] + 21]

                use_WR = False
                use_HR = False

                try:
                    WRVT1 = float(threshold_data[threshold_data[ref_string] == 'WR'].values[0][5])
                    try:
                        WRVT2 = float(threshold_data[threshold_data[ref_string] == 'WR'].values[0][7])
                    except:
                        WRVT2 = 1000
                    use_WR = True
                except:
                    HRVT1 = float(threshold_data[threshold_data[ref_string] == 'HR'].values[0][5])
                    try:
                        HRVT2 = float(threshold_data[threshold_data[ref_string] == 'HR'].values[0][7])
                    except:
                        HRVT2 = 1000
                    use_HR = True
            except:
                pass

            # print('Reading data')
            n_rows = len(ventilatory_data.index)
            # initialise variables
            self.time = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 3,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 3,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 3,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 3,), dtype=np.float32)
            self.load = np.zeros((n_rows - 3,), dtype=np.float32)

            i = 0
            for time_sec in ventilatory_data.t.values[2:-1]:
                # print(i)
                try:
                    self.time[i] = utilities.get_sec(time_sec)
                except:
                    self.time[i] = np.nan

                self.VO2[i] = float(ventilatory_data[ventilatory_data.columns[3]].values[i+2]) * 1000
                self.VCO2[i] = float(ventilatory_data[ventilatory_data.columns[4]].values[i+2]) * 1000

                try:
                    self.HR[i] = float(ventilatory_data[ventilatory_data.columns[7]].values[i+2])
                except:
                    self.HR[i] = 0

                try:
                    self.load[i] = float(ventilatory_data[ventilatory_data.columns[8]].values[i+2])
                except:
                    self.load[i] = 0
                self.VE[i] = float(ventilatory_data[ventilatory_data.columns[11]].values[i+2])
                self.Rf[i] = float(ventilatory_data[ventilatory_data.columns[13]].values[i + 2])
                self.PetO2[i] = float(ventilatory_data[ventilatory_data.columns[14]].values[i + 2])
                self.PetCO2[i] = float(ventilatory_data[ventilatory_data.columns[15]].values[i + 2])

                i += 1

            # self.time = self.time - self.time[0]
            if use_WR:
                VT1_index = np.where(self.load == int(WRVT1))[0][0]
                try:
                    VT2_index = np.where(self.load == int(WRVT2))[0][0]
                except:
                    VT2_index = -1
            elif use_HR:
                VT1_index = np.where(self.HR == int(HRVT1))[0][0]
                try:
                    VT2_index = np.where(self.HR == int(HRVT2))[0][0]
                except:
                    VT2_index = -1

            self.VT1 = self.time[VT1_index]
            self.VT2 = self.time[VT2_index]

        if self.metabolimeter == 'unisbz':

            df = read_csv(self.filename + self.data_extension)

            self.age = float(0)
            # print('Gender')
            self.gender = 'N'
            # print('Weight')
            self.weight = float(0)
            # print('Height')
            self.height = float(0)

            # print('Reading data')
            n_rows = len(df.index)
            self.time = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 2,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.load = np.zeros((n_rows - 2,), dtype=np.float32)

            # get the time in seconds
            # convert time data from "HH:MM:SS" to seconds
            for i in df.index[0:n_rows-2]:
                try:
                    self.time[i] = utilities.get_sec(df[df.columns[0]].values[i])
                    self.VO2[i] = float(df[df.columns[14]].values[i])
                    self.VCO2[i] = float(df[df.columns[15]].values[i])
                    self.VE[i] = float(df[df.columns[11]].values[i])
                    self.HR[i] = float(df[df.columns[3]].values[i])
                    self.PetO2[i] = float(df[df.columns[8]].values[i])
                    self.PetCO2[i] = float(df[df.columns[7]].values[i])
                    self.Rf[i] = float(df[df.columns[6]].values[i])
                    self.load[i] = float(df[df.columns[1]].values[i])
                except:
                    self.time[i] = np.nan
                    self.VO2[i] = np.nan
                    self.VCO2[i] = np.nan
                    self.VE[i] = np.nan
                    self.HR[i] = np.nan
                    self.PetO2[i] = np.nan
                    self.PetCO2[i] = np.nan
                    self.Rf[i] = np.nan
                    self.load[i] = np.nan

        if self.metabolimeter == 'cosmed':

            df = self.df

            # print('Reading age')
            self.age = float(df.values[3, 1])
            # print('Gender')
            try:
                self.gender = df.values[2, 1][0]
            except:
                self.gender = 'M' # default to male
            # print('Weight')
            self.weight = float(df.values[5, 1])
            # print('Height')
            self.height = float(df.values[4, 1])

            # print('Reading data')
            n_rows = len(df.index)
            self.time = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 2,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.load = np.zeros((n_rows - 2,), dtype=np.float32)

            # get the time in seconds
            # convert time data from "HH:MM:SS" to seconds
            for i in df.index[2:n_rows]:
                try:
                    if df.t[0] == 's':
                        try:
                            self.time[i - 2] = int(round((float(df.t.values[i]))*86400))
                        except:
                            self.time[i - 2] = utilities.get_sec(df.t.values[i])
                    else:
                        self.time[i - 2] = utilities.get_sec(df.t.values[i])
                except:
                    self.time[i - 2] = np.nan
                self.VO2[i - 2] = float(df.VO2.values[i])
                self.VCO2[i - 2] = float(df.VCO2.values[i])
                try:
                    self.HR[i - 2] = float(df.HR.values[i])
                except:
                    self.HR[i - 2] = float(df.HF.values[i])
                try:
                    self.Rf[i - 2] = float(df.Rf.values[i])
                except:
                    try:
                        self.Rf[i - 2] = float(df['F.R'].values[i])
                    except:
                        self.Rf[i - 2] = float(df['Af'].values[i])
                try:
                    self.VE[i - 2] = float(df.VE.values[i])
                except:
                    self.VE[i - 2] = float(df.VE_ergo.values[i])
                self.PetO2[i - 2] = float(df.PetO2.values[i])
                self.PetCO2[i - 2] = float(df.PetCO2.values[i])

        if self.metabolimeter == 'cortex_bruce':
            df = read_csv(self.filename + self.data_extension)
            print('Reading data Bruce file!!!!')

            starting_index = df.loc[df[df.columns[0]] == 'Messzeit'].index+2
            n_rows = df[df.columns[0]].last_valid_index()-1

            # print('Reading age')
            my_array = np.where(df[df.columns[6]].notnull() == True, df.index, 0)
            res = next(x for x, val in enumerate(my_array) if val > 0.6)
            try:
                self.age = float(0)
                self.age = 2020 - int(df.iloc[4][2].split('.')[2])
                # print('Gender')
                self.gender = 'N'
                if str(df.iloc[3][2])[0] == 'm':
                    self.gender = 'M'
                if str(df.iloc[3][2])[0] == 'f':
                    self.gender = 'F'
                # print('Weight')
                self.weight = float(0)
                self.weight = float(df.iloc[6][2])
                # print('Height')
                self.height = float(0)
                self.height = float(int(df.iloc[5][2])/100)
            except:
                    self.age = float(0)
                    self.height = float(0)
                    self.gender = 'N'
                    self.weight = float(0)

            self.time = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.HR = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.Rf = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VE = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.load = np.zeros((n_rows - starting_index[0],), dtype=np.float32)

            for i in np.arange(starting_index[0], n_rows - starting_index[0]):
                try:
                    self.time[i - starting_index[0]] = float(utilities.get_sec(df[df.columns[0]].values[i]))
                    self.VO2[i - starting_index[0]] = float(df[df.columns[10]].values[i]) * 1000
                    self.VCO2[i - starting_index[0]] = float(df[df.columns[11]].values[i]) * 1000
                    self.VE[i - starting_index[0]] = float(df[df.columns[4]].values[i])
                    self.HR[i - starting_index[0]] = float(df[df.columns[18]].values[i])
                    self.Rf[i - starting_index[0]] = float(df[df.columns[3]].values[i])
                    self.PetO2[i - starting_index[0]] = float(df[df.columns[8]].values[i])
                    self.PetCO2[i - starting_index[0]] = float(df[df.columns[9]].values[i])
                except:
                    self.time[i - starting_index[0]] = np.nan
                    self.VO2[i - starting_index[0]] = np.nan
                    self.VCO2[i - starting_index[0]] = np.nan
                    self.VE[i - starting_index[0]] = np.nan
                    self.HR[i - starting_index[0]] = np.nan
                    self.Rf[i - starting_index[0]] = np.nan
                    self.PetO2[i - starting_index[0]] = np.nan
                    self.PetCO2[i - starting_index[0]] = np.nan

        if self.metabolimeter == 'cortex_bruce_2':
            df = read_csv(self.filename + self.data_extension)
            print('Reading data Bruce file type 2!!!!')

            starting_index = df.loc[df[df.columns[0]] == 't'].index+2
            n_rows = df[df.columns[0]].last_valid_index()-1

            # print('Reading age')
            my_array = np.where(df[df.columns[6]].notnull() == True, df.index, 0)
            res = next(x for x, val in enumerate(my_array) if val > 0.6)
            try:
                self.age = float(0)
                self.age = 2020 - int(df.iloc[6][2].split('.')[2])
                # print('Gender')
                self.gender = 'N'
                if str(df.iloc[5][2])[0] == 'm':
                    self.gender = 'M'
                if str(df.iloc[5][2])[0] == 'f':
                    self.gender = 'F'
                # print('Weight')
                self.weight = float(0)
                self.weight = float(df.iloc[8][2])
                # print('Height')
                self.height = float(0)
                self.height = float(int(df.iloc[7][2])/100)
            except:
                    self.age = float(0)
                    self.height = float(0)
                    self.gender = 'N'
                    self.weight = float(0)

            self.time = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.HR = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.Rf = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VE = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.load = np.zeros((n_rows - starting_index[0],), dtype=np.float32)

            for i in np.arange(starting_index[0], n_rows - starting_index[0]):
                try:
                    self.time[i - starting_index[0]] = float(utilities.get_sec(df[df.columns[0]].values[i]))
                    self.VO2[i - starting_index[0]] = float(df[df.columns[13]].values[i]) * 1000
                    self.VCO2[i - starting_index[0]] = float(df[df.columns[14]].values[i]) * 1000
                    self.VE[i - starting_index[0]] = float(df[df.columns[7]].values[i])
                    self.HR[i - starting_index[0]] = float(df[df.columns[21]].values[i])
                    self.Rf[i - starting_index[0]] = float(df[df.columns[3]].values[i])
                    self.PetO2[i - starting_index[0]] = float(df[df.columns[11]].values[i])
                    self.PetCO2[i - starting_index[0]] = float(df[df.columns[12]].values[i])
                except:
                    self.time[i - starting_index[0]] = np.nan
                    self.VO2[i - starting_index[0]] = np.nan
                    self.VCO2[i - starting_index[0]] = np.nan
                    self.VE[i - starting_index[0]] = np.nan
                    self.HR[i - starting_index[0]] = np.nan
                    self.Rf[i - starting_index[0]] = np.nan
                    self.PetO2[i - starting_index[0]] = np.nan
                    self.PetCO2[i - starting_index[0]] = np.nan

        if self.metabolimeter == 'mourot':
            df = read_csv(self.filename + self.data_extension)
            # print('Reading data Mourot file!!!!')

            starting_index = df.loc[df[df.columns[0]] == 'Temps'].index+2
            n_rows = df[df.columns[0]].last_valid_index()-1

            # print('Reading age')
            my_array = np.where(df[df.columns[6]].notnull() == True, df.index, 0)
            res = next(x for x, val in enumerate(my_array) if val > 0.6)
            try:
                self.age = float(0)
                self.age = float(df[df.columns[6]][res])
                # print('Gender')
                self.gender = 'N'
                if df[df.columns[18]][res][0] == 'M':
                    self.gender = 'M'
                if df[df.columns[18]][res][0] == 'F':
                    self.gender = 'F'
                # print('Weight')
                self.weight = float(0)
                self.weight = float(df[df.columns[27]][res])
                # print('Height')
                self.height = float(0)
                self.height = float(df[df.columns[22]][res])
            except:
                try:
                    res = res + 2
                    self.age = float(0)
                    self.age = float(df[df.columns[6]][res])
                    # print('Gender')
                    self.gender = 'N'
                    if df[df.columns[18]][res][0] == 'M':
                        self.gender = 'M'
                    if df[df.columns[18]][res][0] == 'F':
                        self.gender = 'F'
                    # print('Weight')
                    self.weight = float(0)
                    self.weight = float(df[df.columns[27]][res])
                    # print('Height')
                    self.height = float(0)
                    self.height = float(df[df.columns[22]][res])
                except:
                    self.age = float(0)
                    self.height = float(0)
                    self.gender = 'N'

            self.time = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.HR = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.Rf = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VE = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.load = np.zeros((n_rows - starting_index[0],), dtype=np.float32)

            for i in np.arange(starting_index[0],n_rows - starting_index[0]):
                try:
                    self.time[i - starting_index[0]] = float(df[df.columns[0]].values[i]) * 60
                    self.VO2[i - starting_index[0]] = float(df[df.columns[9]].values[i]) * 1000
                    self.VCO2[i - starting_index[0]] = float(df[df.columns[11]].values[i]) * 1000
                    self.VE[i - starting_index[0]] = float(df[df.columns[22]].values[i])
                    self.HR[i - starting_index[0]] = float(df[df.columns[27]].values[i])
                    self.PetO2[i - starting_index[0]] = float(df[df.columns[20]].values[i]) * 760
                    self.PetCO2[i - starting_index[0]] = float(df[df.columns[21]].values[i]) * 760
                except:
                    self.time[i - starting_index[0]] = np.nan
                    self.VO2[i - starting_index[0]] = np.nan
                    self.VCO2[i - starting_index[0]] = np.nan
                    self.VE[i - starting_index[0]] = np.nan
                    self.HR[i - starting_index[0]] = np.nan
                    self.PetO2[i - starting_index[0]] = np.nan
                    self.PetCO2[i - starting_index[0]] = np.nan

        if self.metabolimeter == 'mourot_COPD':

            df = read_csv(self.filename + self.data_extension)
            # print('Reading data Mourot file with no age nor BMI data !!!!')

            starting_index = [1]
            n_rows = df[df.columns[0]].last_valid_index() - 3

            # print('Reading age')
            my_array = np.where(df[df.columns[6]].notnull() == True, df.index, 0)
            res = next(x for x, val in enumerate(my_array) if val > 0.6)

            self.age = float(0)
            # print('Gender')
            self.gender = 'N'
            # print('Weight')
            self.weight = float(0)
            # print('Height')
            self.height = float(0)

            self.time = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.HR = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VE = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.load = np.zeros((n_rows - starting_index[0],), dtype=np.float32)

            for i in np.arange(starting_index[0],n_rows):
                try:
                    self.time[i - starting_index[0]] = utilities.get_sec(df[df.columns[0]].values[i])
                except:
                    self.time[i - starting_index[0]] = np.nan
                try:
                    # print(i)
                    self.VO2[i - starting_index[0]] = float(df[df.columns[8]].values[i]) * 1000
                    if df[df.columns[10]].values[i] == ' ':
                        self.VCO2[i - starting_index[0]] = float(df[df.columns[4]].values[i])/float(df[df.columns[12]].values[i]) * 1000
                    else:
                        self.VCO2[i - starting_index[0]] = float(df[df.columns[10]].values[i]) * 1000
                    self.VE[i - starting_index[0]] = float(df[df.columns[4]].values[i])
                    self.HR[i - starting_index[0]] = float(df[df.columns[3]].values[i])
                    self.PetO2[i - starting_index[0]] = float(df[df.columns[15]].values[i]) * 7.60
                    self.PetCO2[i - starting_index[0]] = float(df[df.columns[17]].values[i])
                except:
                    self.time[i - starting_index[0]] = np.nan
                    self.VO2[i - starting_index[0]] = np.nan
                    self.VCO2[i - starting_index[0]] = np.nan
                    self.VE[i - starting_index[0]] = np.nan
                    self.HR[i - starting_index[0]] = np.nan
                    self.PetO2[i - starting_index[0]] = np.nan
                    self.PetCO2[i - starting_index[0]] = np.nan

            self.Rf = self.time * 0

        if self.metabolimeter == 'low':

            df = read_csv(self.filename + self.data_extension)

            # print('Reading age')
            self.age = float(30)
            # print('Gender')
            self.gender = 'N'
            # print('Weight')
            self.weight = float(50)
            # print('Height')
            self.height = float(180)

            # print('Reading data')
            n_rows = len(df.index)
            self.time = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 2,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.load = np.zeros((n_rows - 2,), dtype=np.float32)

            # get the time in seconds
            # convert time data from "HH:MM:SS" to seconds
            for i in df.index[2:n_rows]:
                try:
                    self.time[i - 2] = utilities.get_sec(df[df.columns[0]].values[i])
                    self.VO2[i - 2] = float(df[df.columns[4]].values[i])
                    self.VCO2[i - 2] = float(df[df.columns[5]].values[i])
                    self.VE[i - 2] = float(df[df.columns[3]].values[i])
                    self.HR[i - 2] = float(df[df.columns[2]].values[i])
                except:
                    self.time[i - 2] = np.nan
                    self.VO2[i - 2] = np.nan
                    self.VCO2[i - 2] = np.nan
                    self.VE[i - 2] = np.nan
                    self.HR[i - 2] = np.nan

        if self.metabolimeter == 'VO2Master':

            df = read_csv(self.filename + self.data_extension)

            # print('Reading age')
            self.age = float(30)
            # print('Gender')
            self.gender = 'N'
            # print('Weight')
            self.weight = float(50)
            # print('Height')
            self.height = float(180)

            # print('Reading data')
            n_rows = len(df.index)
            self.time = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 2,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.load = np.zeros((n_rows - 2,), dtype=np.float32)

            # get the time in seconds
            # convert time data from "HH:MM:SS" to seconds
            for i in df.index[1:n_rows]:
                try:
                    self.time[i - 2] = utilities.get_sec(df[df.columns[1]].values[i])
                    self.VO2[i - 2] = float(df[df.columns[3]].values[i])
                    self.VE[i - 2] = float(df[df.columns[6]].values[i])
                    self.Rf[i - 2] = float(df[df.columns[4]].values[i])
                    self.PetO2[i - 2] = float(df[df.columns[8]].values[i]) * 7.60
                except:
                    self.time[i - 2] = np.nan
                    self.VO2[i - 2] = np.nan
                    self.VCO2[i - 2] = np.nan
                    self.VE[i - 2] = np.nan
                    self.HR[i - 2] = np.nan

        if self.metabolimeter == 'centro-monzino':

            df = read_csv(self.filename + self.data_extension)

            # print('Reading age')
            self.age = float(30)
            # print('Gender')
            self.gender = 'N'
            # print('Weight')
            self.weight = float(50)
            # print('Height')
            self.height = float(180)

            # print('Reading data')
            n_rows = len(df.index)
            self.time = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.HR = np.zeros((n_rows - 2,), dtype=np.float32)
            self.Rf = np.zeros((n_rows - 2,), dtype=np.float32)
            self.VE = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - 2,), dtype=np.float32)
            self.load = np.zeros((n_rows - 2,), dtype=np.float32)

            # get the time in seconds
            # convert time data from "HH:MM:SS" to seconds
            for i in df.index[2:n_rows]:
                try:
                    self.time[i - 2] = utilities.get_sec(df[df.columns[0]].values[i])
                    self.VO2[i - 2] = float(df[df.columns[2]].values[i])*1000
                    self.VCO2[i - 2] = float(df[df.columns[3]].values[i])*1000
                    self.VE[i - 2] = float(df[df.columns[4]].values[i])
                    self.HR[i - 2] = float(df[df.columns[9]].values[i])
                    self.PetO2[i - 2] = float(df[df.columns[5]].values[i])
                    self.PetCO2[i - 2] = float(df[df.columns[6]].values[i])
                except:
                    self.time[i - 2] = np.nan
                    self.VO2[i - 2] = np.nan
                    self.VCO2[i - 2] = np.nan
                    self.VE[i - 2] = np.nan
                    self.HR[i - 2] = np.nan

        if self.metabolimeter == 'mourot_cardiac':

            data = read_csv(self.filename + self.data_extension, encoding=result['encoding'], header=0, nrows=0, sep=',')
            d = gender.Detector()

            starting_index = [1]
            n_rows = df[df.columns[0]].last_valid_index() - 3

            # print('Reading age weight and height (neutral gender)')
            test_date = int((data.columns[7].split('/'))[2])
            if test_date < 1900:
                test_date = test_date + 2000

            self.age = test_date - int((data.columns[4].split('/'))[2])
            self.gender = 'N'
            try:
                if d.get_gender(data.columns[1].split(' ')[1].capitalize(), 'france') == 'male':
                    self.gender = 'M'
                if d.get_gender(data.columns[1].split(' ')[1].capitalize(), 'france') == 'female':
                    self.gender = 'F'
            except:
                pass
            self.weight = float(data.columns[10])
            self.height = float(data.columns[13])

            # print('Ventilatory variables')

            self.time = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.HR = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.Rf = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.VE = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.PetCO2 = np.zeros((n_rows - starting_index[0],), dtype=np.float32)
            self.load = np.zeros((n_rows - starting_index[0],), dtype=np.float32)

            for i in np.arange(starting_index[0], n_rows):
                try:
                    self.time[i - starting_index[0]] = float(df[df.columns[0]].values[i]) * 60
                    self.VO2[i - starting_index[0]] = float(df[df.columns[4]].values[i]) * 1000
                    self.VCO2[i - starting_index[0]] = float(df[df.columns[6]].values[i]) * 1000
                    self.VE[i - starting_index[0]] = float(df[df.columns[3]].values[i])
                    self.HR[i - starting_index[0]] = float(df[df.columns[14]].values[i])
                    self.PetO2[i - starting_index[0]] = float(df[df.columns[20]].values[i])
                    self.PetCO2[i - starting_index[0]] = float(df[df.columns[21]].values[i])
                    self.Rf[i - starting_index[0]] = float(df[df.columns[1]].values[i])
                except:
                    self.time[i - starting_index[0]] = np.nan
                    self.VO2[i - starting_index[0]] = np.nan
                    self.VCO2[i - starting_index[0]] = np.nan
                    self.VE[i - starting_index[0]] = np.nan
                    self.HR[i - starting_index[0]] = np.nan
                    self.PetO2[i - starting_index[0]] = np.nan
                    self.PetCO2[i - starting_index[0]] = np.nan
                    self.Rf[i - starting_index[0]] = np.nan

        if self.metabolimeter == 'generated_pyoxynet':
            self.time = df.time
            self.VO2 = df.VO2_I
            self.VCO2 = df.VCO2_I
            self.HR = df.HR_I
            self.Rf = df.RF_I
            self.VE = df.VE_I
            self.PetO2 = df.PetO2_I
            self.PetCO2 = df.PetCO2_I
            self.age = df.Age.mean()
            if df.gender.mean() == 1:
                self.gender = 'F'
            else:
                self.gender = 'M'
            # print('Weight')
            self.weight = df.weight.mean()
            # print('Height')
            self.height = df.height.mean()

        self.time = np.nan_to_num(self.time)
        self.VO2 = np.nan_to_num(self.VO2)
        self.VCO2 = np.nan_to_num(self.VCO2)
        self.HR = np.nan_to_num(self.HR)
        self.Rf = np.nan_to_num(self.Rf)
        self.VE = np.nan_to_num(self.VE)
        self.PetO2 = np.nan_to_num(self.PetO2)
        self.PetCO2 = np.nan_to_num(self.PetCO2)

        if self.metabolimeter != 'low':

            self.Rf_F = utilities.optimal_filter(self.time, self.Rf, 200)
            self.PetO2_F = utilities.optimal_filter(self.time, self.PetO2, 200)
            self.PetCO2_F = utilities.optimal_filter(self.time, self.PetCO2, 200)

        if self.metabolimeter == 'low':
            self.Rf_F = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetO2_F = np.zeros((n_rows - 2,), dtype=np.float32)
            self.PetCO2_F = np.zeros((n_rows - 2,), dtype=np.float32)

        self.VO2_F = utilities.optimal_filter(self.time, self.VO2, 200)
        self.VCO2_F = utilities.optimal_filter(self.time, self.VCO2, 200)
        self.HR_F = utilities.optimal_filter(self.time, self.HR, 200)
        self.VE_F = utilities.optimal_filter(self.time, self.VE, 200)

        # maximal VE value (cutting after)
        max_VE_idx = max([(v, i) for i, v in enumerate(self.VE_F)])[1]
        self.time_F = self.time[:max_VE_idx]
        self.VO2_F = self.VO2_F[:max_VE_idx]
        self.VCO2_F = self.VCO2_F[:max_VE_idx]
        self.HR_F = self.HR_F[:max_VE_idx]
        self.Rf_F = self.Rf_F[:max_VE_idx]
        self.VE_F = self.VE_F[:max_VE_idx]
        self.PetO2_F = self.PetO2_F[:max_VE_idx]
        self.PetCO2_F = self.PetCO2_F[:max_VE_idx]

        self.time_I = np.arange(int(self.time_F[0]), self.time_F[-1])
        self.VO2_I = np.interp(self.time_I, self.time_F, self.VO2_F)
        self.VCO2_I = np.interp(self.time_I, self.time_F, self.VCO2_F)
        self.HR_I = np.interp(self.time_I, self.time_F, self.HR_F)
        self.Rf_I = np.interp(self.time_I, self.time_F, self.Rf_F)
        self.VE_I = np.interp(self.time_I, self.time_F, self.VE_F)
        self.PetO2_I = np.interp(self.time_I, self.time_F, self.PetO2_F)
        self.PetCO2_I = np.interp(self.time_I, self.time_F, self.PetCO2_F)

    def print_details(self):
        if self.gender == "M":
            print("The name of the participant is: " + self.name + '.\n'
                                                                   "He is " + str(
                self.age) + ' years old.\nWeight: ' + str(self.weight) + ' kg\nHeight: ' + str(self.height) + ' cm')
        if self.gender == "F":
            print("The name of the participant is: " + self.name + '.\n'    "She is " + str(
                self.age) + ' years old.\nWeight: ' + str(self.weight) + ' kg\nHeight: ' + str(self.height) + ' cm')

    def create_data_frame(self):

        self.data_frame = pd.DataFrame()

        self.data_frame['time'] = self.time_I
        self.data_frame['VO2_I'] = self.VO2_I
        self.data_frame['VCO2_I'] = self.VCO2_I
        self.data_frame['VE_I'] = self.VE_I
        self.data_frame['HR_I'] = self.HR_I
        self.data_frame['RF_I'] = self.Rf_I
        self.data_frame['PetO2_I'] = self.PetO2_I
        self.data_frame['PetCO2_I'] = self.PetCO2_I

        # variables normalised on VO2
        self.data_frame['VEVO2_I'] = self.VE_I / self.VO2_I
        self.data_frame['VCO2VO2_I'] = self.VCO2_I / self.VO2_I
        self.data_frame['PetO2VO2_I'] = self.PetO2_I / self.VO2_I
        self.data_frame['PetCO2VO2_I'] = self.PetCO2_I / self.VO2_I

        if self.metabolimeter == 'VO2Master':
            # putting to zero
            self.data_frame['VEVCO2_I'] = self.VCO2_I
        else:
            self.data_frame['VEVCO2_I'] = self.VE_I / self.VCO2_I
        self.data_frame['age'] = self.age

        # height in cm
        if self.height > 100:
            self.height = self.height/100
        else:
            pass

        self.data_frame['height'] = np.ones(self.time_I.shape) * self.height
        self.data_frame['weight'] = np.ones(self.time_I.shape) * self.weight

        if self.gender == 'F':
            self.data_frame['gender'] = np.ones(self.time_I.shape)
        else:
            self.data_frame['gender'] = -np.ones(self.time_I.shape)

        try:
            self.data_frame['domain'] = self.domain[:len(self.time_I)]
        except:
            self.data_frame['domain'] = self.time_I

        # compute fitness and age group
        self.data_frame['age_group'] = np.ones(self.time_I.shape) * 2
        self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2
        try:
            if self.age < 40 and self.gender == 'M' or self.gender == 'N':
                self.data_frame['age_group'] = np.ones(self.time_I.shape)
                if max(self.data_frame.VO2_I)/self.weight < 39.5:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 1
                if max(self.data_frame.VO2_I)/self.weight > 48.3:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight >= 39.5 and \
                        max(self.data_frame.VO2_I)/self.weight <= 48.3:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2

            if self.age >= 40 and self.age < 60  and self.gender == 'M' or self.gender == 'N':
                self.data_frame['age_group'] = np.ones(self.time_I.shape) * 2
                if max(self.data_frame.VO2_I)/self.weight < 34.8:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 1
                if max(self.data_frame.VO2_I)/self.weight > 43.3:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight >= 34.8 and \
                        max(self.data_frame.VO2_I)/self.weight <= 43.3:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2

            if self.age >= 60  and self.gender == 'M' or self.gender == 'N':
                self.data_frame['age_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight < 28.8:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 1
                if max(self.data_frame.VO2_I)/self.weight > 36.7:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight >= 28.8 and \
                        max(self.data_frame.VO2_I)/self.weight <= 36.7:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2

            if self.age < 40 and self.gender == 'F' or self.gender == 'N':
                self.data_frame['age_group'] = np.ones(self.time_I.shape)
                if max(self.data_frame.VO2_I)/self.weight < 33.8:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 1
                if max(self.data_frame.VO2_I)/self.weight > 42.4:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight >= 33.8 and \
                        max(self.data_frame.VO2_I)/self.weight <= 42.4:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2

            if self.age >= 40 and self.age < 60  and self.gender == 'F' or self.gender == 'N':
                self.data_frame['age_group'] = np.ones(self.time_I.shape) * 2
                if max(self.data_frame.VO2_I)/self.weight < 32.3:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 1
                if max(self.data_frame.VO2_I)/self.weight > 39.6:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight >= 32.3 and \
                        max(self.data_frame.VO2_I)/self.weight <= 39.6:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2

            if self.age >= 60  and self.gender == 'F' or self.gender == 'N':
                self.data_frame['age_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight < 25.3:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 1
                if max(self.data_frame.VO2_I)/self.weight > 30.6:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 3
                if max(self.data_frame.VO2_I)/self.weight >= 25.3 and \
                        max(self.data_frame.VO2_I)/self.weight <= 30.6:
                    self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2
        except:
            self.data_frame['age_group'] = np.ones(self.time_I.shape) * 2
            self.data_frame['fitness_group'] = np.ones(self.time_I.shape) * 2

    def generate_pickle(self):
        self.data_frame.to_pickle('training_data/' + self.filename_cleared + '.pickle')
        print('Generating pickle file for ' + self.filename)

    def generate_csv(self):
        self.data_frame.to_csv('training_data/' + self.filename_cleared + '.csv')
        print('Generating csv file for ' + self.filename)

    def set_labels(self, label_data_frame):
        import math

        self.domain = np.empty([len(self.time_I),], int)

        if self.metabolimeter == 'cortex':

            self.domain[self.time_I < self.VT1] = -1 # low
            self.domain[(self.time_I >= self.VT1) & (self.time_I < self.VT2)] = 0 # moderate
            self.domain[self.time_I >= self.VT2] = 1 # high

        if self.metabolimeter == 'mourot' or self.metabolimeter == 'mourot_COPD' or self.metabolimeter == 'low' \
                or self.metabolimeter == 'cosmed':

            VT1 = int(label_data_frame[label_data_frame.id==self.filename_cleared].Feature1.values[0])

            if not math.isnan(VT1):
                try:
                    VT2 = int(label_data_frame[label_data_frame.id==self.filename_cleared].Feature2.values[0])
                except:
                    VT2 = self.time_I[-1]

            self.domain[self.time_I < VT1] = -1 # low
            self.domain[(self.time_I >= VT1) & (self.time_I < VT2)] = 0 # moderate
            self.domain[self.time_I >= VT2] = 1 # high

        if self.metabolimeter == 'mourot_cardiac':

            print(self.filename.split('/')[-1])

            VT1 = int(label_data_frame[label_data_frame.id==self.filename.split('/')[-1]].Feature1.values[0])

            # print(VT1)

            if not math.isnan(VT1):
                try:
                    VT2 = int(label_data_frame[label_data_frame.id==self.filename.split('/')[-1]].Feature2.values[0])
                except:
                    VT2 = self.time_I[-1]

            self.domain[self.time_I < VT1] = -1 # low
            self.domain[(self.time_I >= VT1) & (self.time_I < VT2)] = 0 # moderate
            self.domain[self.time_I >= VT2] = 1 # high

        if self.metabolimeter == 'unisbz':

            VT1 = int(label_data_frame[label_data_frame.id==int(self.filename_cleared)].Feature1.values[0])

            if not math.isnan(VT1):
                try:
                    VT2 = int(label_data_frame[label_data_frame.id==int(self.filename_cleared)].Feature2.values[0])
                except:
                    VT2 = self.time_I[-1]

            self.domain[self.time_I < VT1] = -1 # low
            self.domain[(self.time_I >= VT1) & (self.time_I < VT2)] = 0 # moderate
            self.domain[self.time_I >= VT2] = 1 # high