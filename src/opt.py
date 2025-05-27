import os
import random
from ase.io import read, Trajectory
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from deepmd.calculator import DP



def optimize_structure(poscar_path, fmax, opt_stepi, model):
    calc = DP(model=model, head='DOWNSTREAM_DATA')  # initialize DP calculator in the child process

    #stress_lo_t = int(stress_lo / 10)
    #stress_hi_t = int(stress_hi / 10)
    #pstress = random.randint(stress_lo_t, stress_hi_t) * 10
    #aim_stress = 1.0 * pstress * 0.006242
    #print(f'selected pressure: {pstress} GPa {stress_lo} {stress_hi}')

    to_be_opti = read(poscar)
    to_be_opti.calc = calc
    ucf = UnitCellFilter(to_be_opti, scalar_pressure=aim_stress)
    # opt
    traj = Trajectory('traj.{}'.format(poscar.split('.')[-1]), 'w', to_be_opti)
    opt = LBFGS(ucf)
    opt.attach(traj.write, interval=1)
    try:
        opt.run(fmax=fmax,steps=opt_step)
    except:
        print('Something went wrong, perhaps OOM please check')



