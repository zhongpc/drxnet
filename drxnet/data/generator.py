import json
import numpy as np
import os
from pymatgen.core import Composition
from copy import deepcopy
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import pchip_interpolate
from scipy import signal

global composition_list;
composition_list = ['Li', 'Mg', 'Al',
                    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                    'Zr', 'Nb', 'Mo', 'W',
                    'O', 'F']

global compositionMass_list;
compositionMass_list = [6.94, 24.305, 26.982,
                        47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693,
                        91.223, 92.906, 95.95, 183.84,
                        15.999, 18.998]

def generate_composition_vector(comp_dict):
    comp_vec = np.zeros(len(composition_list))
    comp_mass = 0
    for key in comp_dict.keys():
        formula_unit = comp_dict[key]
        specie_index = composition_list.index(key)
        comp_vec[specie_index] = formula_unit / 2
        comp_mass += compositionMass_list[specie_index] * formula_unit
    num_Li = comp_dict['Li']
    return comp_vec, comp_mass, num_Li

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


class convert_csvData_Q0_Q:
    def __init__(self, std_data, maxCycleNumber=1000, cdc_info='discharge_list'):
        """
        Generate tensor for machine learning input.
        The result will be saved as a numpy array to be read

        info: contains all cycling data from experimental measurements
        """

        self.comp_dict = std_data['composition']
        self.composition = Composition.from_dict(self.comp_dict)
        self.test_current = std_data['test_current(mA/g)']
        self.low_V = std_data['low_voltage(V)']
        self.high_V = std_data['high_voltage(V)']
        self.total_cycle_num = len(std_data[cdc_info])
        self.profileInformation = std_data[cdc_info]

        self.maxCycleNumber = maxCycleNumber

        comp_vector, comp_mass, num_Li = generate_composition_vector(self.comp_dict)
        self.theory_Q = num_Li * 96485 / (3.6 * comp_mass)


    def write_lines(self, out_path, sample_ratio = 0.1, init_count = 0):
        """
        only generate the start and end of charge/discharge profile
        :return:
        """

        total_cycle_num = len(self.profileInformation)
        f = open(out_path, 'w')
        save_line  = "material_id,composition,V_low,V_high,rate,cycle,theory_Q,Vii,Q0ii,Qii,dQdVii\n"
        f.writelines(save_line)
        count = deepcopy(init_count)



        for _info in self.profileInformation:
            cycle_number = _info['cycle_num']

            V_std = np.array(_info['V_std'])
            Q0_std = np.array(_info['Q0_std'])
            Q_std = np.array(_info['Q_std'])
            dQdV_std = np.array(_info['dQdV_std'])

            sample_step = int( 1 / sample_ratio)

            if (cycle_number > self.maxCycleNumber):
                f.close()
                break
                # ignore the profile when the number is larger than the limit

            for ii in range(0, V_std.shape[0], sample_step):
                Vii = V_std[ii]
                Q0ii = Q0_std[ii]
                Qii = Q_std[ii]
                dQdVii = dQdV_std[ii]

                if Vii < self.low_V:
                    Vii = self.low_V
                elif Vii > self.high_V:
                    Vii = self.high_V

                if Q0ii < 0:
                    Q0ii = 0
                if Qii < 0:
                    Qii = 0
                if dQdVii > 0:  # dQdV is always negative
                    dQdVii = 0


                if Q0ii > 360:
                    print(Q0ii, out_path)

                # current_index = encode_rate(self.test_current)

                count += 1
                save_line = '{count},{composition},{low_V:.2f},{high_V:.2f},{current_index},{cycle_number},{theory_Q:.3f},{Vii:.3f},{Q0ii:.3f},{Qii:.3f},{dQdVii:.3f}\n'.format(
                             count = count, composition = self.composition,
                             low_V = self.low_V, high_V = self.high_V,
                             current_index = self.test_current, cycle_number = cycle_number, theory_Q = self.theory_Q,
                             Vii = Vii, Q0ii = Q0ii, Qii = Qii,  dQdVii = dQdVii
                             )

                f.writelines(save_line)

        f.close()

        return count


class convert_csvData_train_val:
    def __init__(self, std_data, maxCycleNumber=1000, cdc_info='discharge_list'):
        """
        Generate tensor for machine learning input.
        The result will be saved as a numpy array to be read

        info: contains all cycling data from experimental measurements
        """
        self.composition = Composition.from_dict(std_data['composition'])
        self.test_current = std_data['test_current(mA/g)']
        self.low_V = std_data['low_voltage(V)']
        self.high_V = std_data['high_voltage(V)']
        self.total_cycle_num = len(std_data[cdc_info])
        self.profileInformation = std_data[cdc_info]

        self.maxCycleNumber = maxCycleNumber

        comp_vector, comp_mass, num_Li = generate_composition_vector(std_data['composition'])
        self.theory_Q = num_Li * 96485 / (3.6 * comp_mass)


    def write_lines(self, train_path, val_path, sample_ratio = 0.1, init_count = 0, init_val_count = 0, val_ratio = 0.1):
        """
        only generate the start and end of charge/discharge profile
        :return:
        """

        total_cycle_num = len(self.profileInformation)
        f = open(train_path, 'w')
        f_val = open(val_path, 'w')

        save_line = "material_id,composition,V_low,V_high,rate,cycle,theory_Q,Vii,Q0ii,Qii,dQdVii\n"
        f.writelines(save_line)
        f_val.writelines(save_line)

        count = deepcopy(init_count)
        val_count = deepcopy(init_val_count)

        cycle_indices = np.arange(2,len(self.profileInformation) + 1)
        val_indices = np.random.choice(cycle_indices, int(val_ratio * len(self.profileInformation)),replace  = False)

        for _info in self.profileInformation:
            cycle_number = _info['cycle_num']

            V_std = np.array(_info['V_std'])
            Q0_std = np.array(_info['Q0_std'])
            Q_std = np.array(_info['Q_std'])
            dQdV_std = np.array(_info['dQdV_std'])

            sample_step = int( 1 / sample_ratio)

            if (cycle_number > self.maxCycleNumber):
                f.close()
                break
                # ignore the profile when the number is larger than the limit

            for ii in range(0, V_std.shape[0], sample_step):
                Vii = V_std[ii]
                Q0ii = Q0_std[ii]
                Qii = Q_std[ii]
                dQdVii = dQdV_std[ii]

                if Vii < self.low_V:
                    Vii = self.low_V
                elif Vii > self.high_V:
                    Vii = self.high_V

                if Q0ii < 0:
                    Q0ii = 0
                if Qii < 0:
                    Qii = 0
                if dQdVii > 0:  # dQdV is always negative
                    dQdVii = 0


                if Q0ii > 360:
                    print(Q0ii, train_path)

                if cycle_number in val_indices:
                    save_line = '{count},{composition},{low_V:.2f},{high_V:.2f},{current_index},{cycle_number},{theory_Q:.3f},{Vii:.3f},{Q0ii:.3f},{Qii:.3f},{dQdVii:.3f}\n'.format(
                             count = val_count, composition = self.composition,
                             low_V = self.low_V, high_V = self.high_V,
                             current_index = self.test_current, cycle_number = cycle_number, theory_Q = self.theory_Q,
                             Vii = Vii, Q0ii = Q0ii, Qii = Qii,  dQdVii = dQdVii
                             )

                    f_val.writelines(save_line)
                    val_count += 1
                else:
                    save_line = '{count},{composition},{low_V:.2f},{high_V:.2f},{current_index},{cycle_number},{theory_Q:.3f},{Vii:.3f},{Q0ii:.3f},{Qii:.3f},{dQdVii:.3f}\n'.format(
                             count = val_count, composition = self.composition,
                             low_V = self.low_V, high_V = self.high_V,
                             current_index = self.test_current, cycle_number = cycle_number, theory_Q = self.theory_Q,
                             Vii = Vii, Q0ii = Q0ii, Qii = Qii,  dQdVii = dQdVii
                             )

                    f.writelines(save_line)
                    count += 1

        f.close()
        f_val.close()

        return count, val_count


class convert_csvData:
    def __init__(self, std_data, maxCycleNumber=1000, cdc_info='discharge_list'):
        """
        Generate tensor for machine learning input.
        The result will be saved as a numpy array to be read

        info: contains all cycling data from experimental measurements
        """
        self.composition = Composition.from_dict(std_data['composition'])
        self.test_current = std_data['test_current(mA/g)']
        self.low_V = std_data['low_voltage(V)']
        self.high_V = std_data['high_voltage(V)']
        self.total_cycle_num = len(std_data[cdc_info])
        self.profileInformation = std_data[cdc_info]

        self.maxCycleNumber = maxCycleNumber


    def write_lines(self, out_path, sample_ratio = 0.1, init_count = 0):
        """
        only generate the start and end of charge/discharge profile
        :return:
        """

        total_cycle_num = len(self.profileInformation)
        f = open(out_path, 'w')
        save_line  = "material_id,composition,V_low,V_high,rate,cycle,Vii,Q0ii,DQii,dQdVii\n"
        f.writelines(save_line)
        count = deepcopy(init_count)



        for _info in self.profileInformation:
            cycle_number = _info['cycle_num']

            V_std = np.array(_info['V_std'])
            Q0_std = np.array(_info['Q0_std'])
            DQ_std = np.array(_info['DQ_std'])
            # Q_std = np.array(_info['Q_std'])
            dQdV_std = np.array(_info['dQdV_std'])

            sample_step = int( 1 / sample_ratio)

            if (cycle_number > self.maxCycleNumber):
                f.close()
                break
                # ignore the profile when the number is larger than the limit

            for ii in range(0, V_std.shape[0], sample_step):
                Vii = V_std[ii]
                Q0ii = Q0_std[ii]
                DQii = DQ_std[ii]
                dQdVii = dQdV_std[ii]

                if Vii < self.low_V:
                    Vii = self.low_V
                elif Vii > self.high_V:
                    Vii = self.high_V

                if Q0ii < 0:
                    Q0ii = 0
                if dQdVii > 0:  # dQdV is always negative
                    dQdVii = 0


                if Q0ii > 360:
                    print(Q0ii, out_path)

                # current_index = encode_rate(self.test_current)

                count += 1
                save_line = '{count},{composition},{low_V:.2f},{high_V:.2f},{current_index},{cycle_number},{Vii:.3f},{Q0ii:.3f},{DQii:.3f},{dQdVii:.3f}\n'.format(
                             count = count, composition = self.composition,
                             low_V = self.low_V, high_V = self.high_V,
                             current_index = self.test_current, cycle_number = cycle_number,
                             Vii = Vii, Q0ii = Q0ii, DQii = DQii, dQdVii = dQdVii
                             )

                f.writelines(save_line)

        f.close()

        return count
