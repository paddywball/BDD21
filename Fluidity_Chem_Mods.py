import numpy as np
from scipy.optimize import fsolve
import scipy.constants as const

"""
Class of modules used in Fluidity_Optimize_Parallel_Example.py
Author: Patrick Ball.
"""

class Fluidity_Chem:

    """
    Class contains parameters and methods required to calculate melt compositions
    along melt fraction profiles.
    
    Melting paths calculated according to the parameterisation of Katz et al.
    (2003, G-cubed) using code reproduced by Fergus McNab and Patrick Ball.

    Code to calculate modal mineralogy, melt stoichiometry, partition coefficients,
    and therefore melt composition, developed by Patrick Ball.
    """

    def __init__(self, bulk_H2O, mod_cpx, A1, A2, A3, B1, B2, B3, C1, C2, C3, r1, r2,
                 beta1, beta2, K, gamma, D_H2O, chi1, chi2, lam, grav, heat_cap, expans_s,
                 expans_f, density_s, density_f, melt_entropy, dP, Ph_Cov, X_inc,
                 h_inc, X_0, spl_out, plg_in, gnt_out, spl_in, Fn_spl, Fn_gnt):
        
        ##########################################
        # Parameters to calculate melt paths.
        # Unless specified, all constants are from
        # Katz et al., (2003) - G-cubed.
        ##########################################
         
        self.bulk_H2O=0.  # Bulk water content (wt%).

        # mod_cpx updated here as described in the Supplementary Materials.
        self.mod_cpx=0.18 # Modal clinopyroxene.

        # Coefficients used to define solidus.
        self.A1=1085.7 # oC.
        self.A2=132.9 # oC GPa(^-1).
        self.A3=-5.1 # oC GPa(^-2).

        # B1 updated here as described in the Supplementary Materials.
        # Coefficients used to define lherzolite liquidus.
        self.B1=1520 # oC.
        self.B2=80. # oC GPa(^-1).
        self.B3=-3.2 # C GPa(^-2).

        # Coefficients used to define bulk liquidus.
        self.C1=1780. # oC.
        self.C2=45. # oC GPa(^-1).
        self.C3=-2. # oC GPa(^-2).

        # r1 and r2 updated here as described in the Supplementary Materials.
        # Coefficients used to calculate reaction coefficient of cpx.
        self.r1=0.94 # cpx/melt.
        self.r2=-0.1 # cpx/melt GPa(^-1).

        # beta2 updated here as described in the Supplementary Materials.
        # Coefficients used to calculate melt fraction.
        self.beta1=1.5
        self.beta2=1.2

        # Coefficients used to calculate hydrous melt correction.
        self.K=43. # oC wt%^(-gamma).
        self.gamma=0.75

        # Partition coefficient of water. Assumed similar to Ce.
        self.D_H2O=0.01

        # Coefficients used to calculate water saturation.
        self.chi1=12. # wt% GPa^(-lam).
        self.chi2=1. # wt% GPa^(-1).
        self.lam=0.6

        # Some material properties.
        self.grav=const.g # Gravity, m s^(-2)
        # Specific Heat Capacity and Thermal expansivity of the solid
        # are from Shorttle et al., (2014) - EPSL
        self.heat_cap=1187. # Specific heat capactiy, J kg^(-1) K^(-1).
        self.expans_s=30.*(10.**-6.) # Thermal expansivity of solid, K^(-1).
        self.expans_f=68.*(10.**-6.) # Thermal expansivity of fluid, K^(-1).
        self.density_s=3300. # STP density of solid, kg m^(-3).
        self.density_f=2900. # STP density of fluid, kg m^(-3).
        # Entropy of melting from Shorttle et al., (2014) - EPSL
        self.melt_entropy=407. # Entropy of melting, J kg^(-1) K^(-1).

        # Pressure increment.
        self.dP = -0.0001 # GPa

#-------------------------------------------------------------

        ###################
        # Parameters to calculate melt composition.
        ###################

        # Constant for Pressure to Depth Conversion (km GPa^-1)
        self.Ph_Cov = 31.4 # (km GPa^-1)
        #self.Ph_Cov = 32.37 # (km GPa^-1)

        # Increment the algorithm steps in to calculate composition along the melt path.
        self.X_inc = 0.0001
        # Increment in depth space.
        self.h_inc = 1. # (km)
        # Initial melt fraction.
        self.X_0 = 0.

        # Depth of Phase Transitions
        self.spl_out = 25 # Depth above which spinel is no longer stable (km)
        self.plg_in = 35 # Depth above which plagioclase is stable (km)
        self.gnt_out = 63 # Depth above which garnet is no longer stable (km)
        self.spl_in = 72 # Depth above which spinel is stable (km)

        # List of initial mineral proportions in the lherzolite source (Fn).
        # Listed in order: ol, opx, cpx, plg, spl, gnt.
        self.Fn_spl = [0.578, 0.27, 0.119, 0., 0.033, 0.]
        self.Fn_gnt = [0.598, 0.211, 0.0758, 0., 0., 0.115]
        
#-------------------------------------------------------------

    def calc_water_in_melt(self, F):

        """
        Calculate concentration of water in melt at given melt fraction, F.
        EQ. 18, Katz et al. (2003, G-cubed.)
        """

        return self.bulk_H2O / ( self.D_H2O + F*(1. - self.D_H2O) )

    def calc_hydrous_correction(self, F):

        """
        Calculate correction to solidus and liquidi for hydrous melting. K.
        EQ. 16, Katz et al. (2003, G-cubed.)
        """

        water_in_melt = self.calc_water_in_melt(F)

        return self.K * (water_in_melt**self.gamma)

    def calc_anhydrous_solidus(self, P):

        """
        Returns anhydrous solidus temperature at given pressure(s) in GPa.
        EQ 4., Katz et al. (2003, G-cubed.)
        """

        return self.A1 + self.A2*P + self.A3*(P**2.)

    def calc_anhydrous_lherzolite_liquidous(self, P):

        """
        Returns anhydrous lherzolite (i.e. cpx present) liquidous at given pressure(s) in GPa.
        EQ 5., Katz et al. (2003, G-cubed.)
        """

        return self.B1 + self.B2*P + self.B3*(P**2.)

    def calc_anhydrous_bulk_liquidous(self, P):

        """
        Returns anhydrous bulk liquidous at given pressure(s) in GPa.
        EQ 10., Katz et al. (2003, G-cubed).
        """

        return self.C1 + self.C2*P + self.C3*(P**2.)

    def calc_solidus(self, P):

        """
        Returns solidus temperature at given pressure(s) in GPa.
        EQ 4., Katz et al. (2003, G-cubed).
        """

        return self.calc_anhydrous_solidus(P) - self.calc_hydrous_correction(0.)

    def calc_dTdP_solidus(self, P):

        """
        Returns gradient with respect to pressure of solidus at given
        pressure(s) in GPa.
        Differentiate EQ 4., Katz et al. (2003, G-cubed).
        """

        return self.A2 + 2.*self.A3*P

    def calc_dTdP_lherzolite_liquidous(self, P):

        """
        Returns gradient with respect to pressure of lherzolite liquidus at given
        pressure(s) in GPa.
        Differentiate EQ 5., Katz et al. (2003, G-cubed).
        """

        return self.B2 + 2.*self.B3*P

    def calc_dTdP_bulk_liquidous(self, P):

        """
        Returns gradient with respect to pressure of bulk liquidus at given
        pressure(s) in GPa.
        Differentiate EQ 5., Katz et al. (2003, G-cubed).
        """

        return self.C2 + 2.*self.C3*P

    def calc_adiabat(self, P, Tp):

        """
        Returns adiabatic temperature for given potential temperature in oC
        at given pressure(s) in GPa.

        Expression comes from integrating EQ. 23 with F = 0 to obtain T(P).
        Requires some rearranging first. See also e.g. McKenzie and Bickle
        (1989, JoP).
        """

        return const.convert_temperature(Tp, 'Celsius', 'Kelvin') * np.exp( (self.expans_s*(P * const.giga))/(self.density_s * self.heat_cap)) - 273.15


    def find_solidus_adiabat_intersection(self, Tp):

        """
        Returns intersection of the solidus and the adiabat for a given potential
        temperature.

        Note: Since the quadratic function is used for the solidus has a turning point
        around 12 GPa, 2000 oC, Tp
        """

        # define function for use in fsolve.
        #   - reads P, T, Tp, returns array with results from adiabat and solidus.
        #   - aims to find P and T which give [T - func(P, Tp) = 0] for both.
        def f(PT, Tp):
            P, T = PT
            z = np.array( [ T - self.calc_adiabat(P, Tp),
                            T - (self.calc_solidus(P)) ] )
            return z

        # try fsolve. if failing return nans.
        try:
            P_intercept, T_intercept = fsolve(f, [0., Tp], args=Tp)
        except RuntimeWarning:
            P_intercept = np.nan
            T_intercept = np.nan

        return P_intercept, T_intercept

    def calc_cpx_out_fraction(self, P):

        """
        Returns melt fraction at which cpx is exhausted for a given pressure in GPa.
        EQ. 6 & EQ. 7, Katz et al. (2003, G-cubed).
        """

        return self.mod_cpx / (self.r1 + self.r2*P)

    def calc_cpx_out_temperature(self, P):

        """
        Returns temperature at which cpx is exhausted for a given pressure in GPa.
        EQ. 9, Katz et al. (2003, G-cubed).
        """

        F_cpx_out = self.calc_cpx_out_fraction(P)
        T_lherz_liq = self.calc_anhydrous_lherzolite_liquidous(P)
        T_sol = self.calc_anhydrous_solidus(P)

        return (F_cpx_out**(1./self.beta1))*(T_lherz_liq - T_sol) + T_sol

    def calc_dTdF_P(self, P, F, cpx_present=True):

        """
        Calculates gradient of temperature with respect to melt fraction at constant
        pressure.

        Modified version of EQ. 21 to include hydrous parts. Comes from
        rearranging and differentiating EQ. 19, Katz et al. (2003, G-cubed).
        """

        if F == 0.:

            return 0.

        else:

            beta1 = self.beta1
            beta2 = self.beta2
            K = self.K
            bulk_H2O = self.bulk_H2O
            D_H2O = self.D_H2O
            gamma = self.gamma
            anhydrous_lherz_liq = self.calc_anhydrous_lherzolite_liquidous(P)
            anhydrous_liq = self.calc_anhydrous_bulk_liquidous(P)
            anhydrous_sol = self.calc_anhydrous_solidus(P)
            F_cpx_out = self.calc_cpx_out_fraction(P)
            T_cpx_out = self.calc_cpx_out_temperature(P)

            # dry melting
            if F < F_cpx_out:
                # (EQ. 21). Rearranging and differentiating EQ. 19.
                dTdF_P = (1./beta1) * (F**((1.-beta1)/beta1)) * (anhydrous_lherz_liq - anhydrous_sol)
            else:
                # Rearranging and differentiating EQ. 8.
                dTdF_P = anhydrous_liq - T_cpx_out
                dTdF_P /= beta2 * (1.-F_cpx_out)
                dTdF_P *= ( (F - F_cpx_out)/(1. - F_cpx_out) )**((1. - beta2)/beta2)

            # wet melting correction
            dTdF_P += K * gamma * (bulk_H2O**gamma) * (1. - D_H2O) / ( (D_H2O + F*(1.-D_H2O))**(gamma+1.) )

            return dTdF_P

    def calc_dTdP_F(self, P, F, cpx_present=True):

        """
        Calculates gradient of temperature with respect to pressure at constant
        melt fraction.

        EQ. 22, Katz et al. (2003, G-cubed). Note typo (I think) in the paper -
        F*beta should be F**(1./beta1). Comes from rearranging and differentiating
        EQ. 19.
        """

        beta1           = self.beta1
        beta2           = self.beta2
        r1              = self.r1
        r2              = self.r2
        mod_cpx         = self.mod_cpx
        dTdP_bulk_liq   = self.calc_dTdP_bulk_liquidous(P)
        dTdP_lherz_liq  = self.calc_dTdP_lherzolite_liquidous(P)
        dTdP_sol        = self.calc_dTdP_solidus(P)
        anhydrous_lherz_liq = self.calc_anhydrous_lherzolite_liquidous(P)
        anhydrous_liq = self.calc_anhydrous_bulk_liquidous(P)
        anhydrous_sol = self.calc_anhydrous_solidus(P)
        F_cpx_out = self.calc_cpx_out_fraction(P)
        T_cpx_out = self.calc_cpx_out_temperature(P)

        if F < F_cpx_out:
            dTdP_F = (F**(1./beta1)) * (dTdP_lherz_liq - dTdP_sol) + dTdP_sol
        else:
            dTdP_F = (F_cpx_out**(1./beta1)) * (dTdP_lherz_liq - dTdP_sol)
            dTdP_F -= (r2/beta1) * (mod_cpx**(1./beta1)) * ((r1 + r2*P)**(-(1.+beta1)/beta1)) * (anhydrous_lherz_liq - anhydrous_sol)
            dTdP_F += dTdP_sol
            dTdP_F += (( dTdP_bulk_liq - dTdP_sol - (F_cpx_out**(1./beta2)) * (dTdP_lherz_liq - dTdP_sol) +
                         (r2/beta2) * (mod_cpx**(1./beta2)) * ((r1 + r2*P)**(-(1.+beta2)/beta2)) * (anhydrous_lherz_liq - anhydrous_sol) ) *
                         ((F - F_cpx_out) / (1. - F_cpx_out))**(1./beta2) )
            dTdP_F += (( anhydrous_liq - T_cpx_out ) * ( (mod_cpx*r2) / (beta2*((r1+r2*P)**2.)) ) *
                          ((F - F_cpx_out)**((1.-beta2)/beta2))*((1. - F_cpx_out)**(-1./beta2)) -  ((F - F_cpx_out)**(1./beta2))*((1. - F_cpx_out)**(-(1.+beta2)/beta2)) )

        return dTdP_F

    def calc_dFdP_S(self, P, F, args):

        """
        Calculates gradient of melt fraction path with respect to pressure at constant
        entropy.

        Note typo in paper - (1-F) term should use properties of solid.

        Requires dTdP_F and dTdF_P.

        EQ. 20, Katz et al. (2003, G-cubed).

        See also EQ. D7, McKenzie (1984, JoP).
        """

        T = const.convert_temperature(args[0], 'Celsius', 'Kelvin')
        cpx = args[1]

        dTdP_F = self.calc_dTdP_F(P, F, cpx_present=cpx)
        dTdF_P = self.calc_dTdF_P(P, F, cpx_present=cpx)

        dFdP_S = -(self.heat_cap / T) * dTdP_F
        dFdP_S += F * ((self.expans_f/(10.**-6.)) / (self.density_f/1000.))
        dFdP_S += (1.-F) * ((self.expans_s/(10.**-6.)) / (self.density_s/1000.))
        dFdP_S /= self.melt_entropy + (self.heat_cap / T) * dTdF_P

        return dFdP_S

    def calc_dTdP_S(self, P, T, args):

        """
        Calculates gradient of temperature with respect to pressure at constant
        entropy.

        Requires dFdP_S.

        EQ. 23, Katz et al. (2003, G-cubed).
        """

        F = args[0]
        cpx = args[1]

        dFdP_S = self.calc_dFdP_S(P, F, (T, cpx))

        dTdP_S = F * ((self.expans_f/(10.**-6.)) / (self.density_f/1000.))
        dTdP_S += (1. - F) * ((self.expans_s/(10.**-6.)) / (self.density_s/1000.))
        dTdP_S -= self.melt_entropy * dFdP_S
        dTdP_S *= const.convert_temperature(T, 'Celsius', 'Kelvin') / self.heat_cap

        return dTdP_S

    def runge_kutta(self, function, P, y0, args):

        """
        Use fourth-order Runge Kutta scheme to evaluate value of y (i.e.
        temperature or melt fraction) at position P, given value at previous
        position, y0.

        i.e. solve equation of the form dy/dP = f(P) for y.
        """

        k1 = function( P,               y0,                  args )
        k2 = function( P + 0.5*self.dP, y0 + 0.5*self.dP*k1, args )
        k3 = function( P + 0.5*self.dP, y0 + 0.5*self.dP*k2, args )
        k4 = function( P + self.dP,     y0 + self.dP*k3,     args )

        return y0 + (self.dP/6.) * (k1 + 2.*k2 + 2*k3 + k4)

    def melting_solver(self, P, T_in, F_in, cpx_present=True):

        """
        Given values of pressure, temperature and melt fraction, return
        temperature and melt fraction at next pressure increment.
        """

        F_out = self.runge_kutta(self.calc_dFdP_S, P, F_in, (T_in, cpx_present))
        T_out = self.runge_kutta(self.calc_dTdP_S, P, T_in, (F_in, cpx_present))

        return T_out, F_out

#-------------

    def calc_PTX(self, Tp, P, T, X_full):

        """
        Generates regularly spaced arrays of Pressure (P), Temperature (T) and Melt Fraction (X)
        from original arrays of P, T, X. 
        
        X array is in regular intervals of X_inc and corresponding values of P and T are at same
        indexes
        
        If original array of P, T, X begins above the solidus, a melt path from the solidus to 
        the start of the original array is appended to the original array to form the output array.
        This prefix melt path assumes a potential temperature.
        """

        X_inc = self.X_inc
        X_0 = self.X_0

        P_0, T_0 = self.find_solidus_adiabat_intersection(Tp)

        # Add solidus intersect to melt path
        P = np.append(P_0, P)
        T = np.append(T_0, T)
        X_full = np.append(X_0, X_full)

        # Extra melt fraction step to finish Runge Kutta scheme.
        X_rk = X_full[-1] + X_inc
        # Generate Melt Path at X_inc Intervals                
        X = np.arange(X_0, (X_rk+X_inc), X_inc)

        # Generate P T and Depth, h, Paths Corresponding to Regular Intervals of X_inc.
        T = np.interp(X, X_full, T) # oC
        P = np.interp(X, X_full, P) # GPa

        # Convert to Kelvin
        T = const.convert_temperature(T, 'Celsius', 'Kelvin')

        return P, T, X

#-------------

    def calc_PTX_h(self, P, T, X):

        """
        Calculates X as a function of Depth, h, in km.
        Outputs both X and h in 1km depth intervals -> X_km and h_km.
        NEED TO LOOK AT WHAT THIS IS USED FOR!!!!!!!
        """

        Ph_Cov = self.Ph_Cov
        h_inc = self.h_inc

        # Check melt column reaches the surface
        h_top = P[-1] * Ph_Cov

        if h_top < self.gnt_out:

            # Extend melt path to the surface through adiabatic decompression

            P_top = np.arange(P[-1], 0., self.dP)        
            T_top = np.zeros_like(P_top)
            T_top[0] = T[-1]
            X_top = np.zeros_like(P_top)
            X_top[0] = X[-1]

            for i, Pi in enumerate(P_top):

                # Skip ahead on first iteration
                if i==0:
                    continue

                # Check whether or not cpx is exhausted.
                F_cpx_out = self.calc_cpx_out_fraction(Pi)

                # Run solver to evaluate temperature and melt fraction at
                # next pressure value.
                if X_top[i-1] < F_cpx_out:
                    T_top[i], X_top[i] = self.melting_solver(Pi, T_top[i-1], X_top[i-1])

                    if self.calc_water_in_melt(X_top[i]) > (self.chi1*(Pi**self.lam) + self.chi2*Pi):
                        print("WARNING: water saturated")

                else:
                    T_top[i], X_top[i] = self.melting_solver(Pi, T_top[i-1], X_top[i-1], cpx_present=False)

            P_all = np.append(P, P_top)
            X_all = np.append(X, X_top)
        
        else:

            P_all = P
            X_all = X
        
        # Convert from pressure (P) to depth (h)
        # Write to h_km array from 0km to base of melt column at intervals of h_inc.
        h = - P * Ph_Cov
        h_all = - P_all * Ph_Cov
        h_dash = int(h_all[-1])
        h_0 = int(h_all[0])
        h_km = np.arange(h_0, (h_dash + h_inc), h_inc)

        # Interpolate X to find values of X at each 1km depth interval (X_km).
        X_km = np.interp(h_km, h_all, X_all)

        # Reverse Depth so decreases as pressure increases
        h = h * -1.
        h_km = h_km * -1.

        return h, h_km, h_dash, X_km

#-------------

    def phase_X(self, X_km, h_km, h_dash):

        """
        Calculates X at which each phase transition occurs.        
        """        

        spl_out = self.spl_out
        plg_in = self.plg_in
        gnt_out = self.gnt_out
        spl_in = self.spl_in
        h_inc = self.h_inc

        if ((spl_out <= h_km[0]) & (spl_out >= h_km[-1])):
            X_spl_out = X_km[h_dash - int(spl_out / h_inc) - 1]
        elif (spl_out > h_km[0]):
            X_spl_out = -1.
        elif(spl_out < h_km[-1]):
            X_spl_out = 1.
            
        if ((plg_in <= h_km[0]) & (plg_in >= h_km[-1])):
            X_plg_in = X_km[h_dash - int(plg_in / h_inc) - 1]
        elif (plg_in > h_km[0]):
            X_plg_in = -1.
        elif(plg_in < h_km[-1]):
            X_plg_in = 1.
            
        if ((gnt_out <= h_km[0]) & (gnt_out >= h_km[-1])):
            X_gnt_out = X_km[h_dash - int(gnt_out / h_inc) - 1]
        elif (gnt_out > h_km[0]):
            X_gnt_out = -1.
        elif(gnt_out < h_km[-1]):
            X_gnt_out = 1.
            
        if ((spl_in <= h_km[0]) & (spl_in >= h_km[-1])):
            X_spl_in = X_km[h_dash - int(spl_in / h_inc) - 1]
        elif (spl_in > h_km[0]):
            X_spl_in = -1.
        elif(spl_in < h_km[-1]):
            X_spl_in = 1.

        return X_spl_out, X_plg_in, X_gnt_out, X_spl_in

#-------------

    def calc_polynomials(self, X, P, eNd):
        """
        Calculates modal mineralogy of each mineral and melt stoicheometry as a function of X and P
        using second order polynomial equations.
        """
        
        X_inc = self.X_inc
        
        # Name of each mineral calculated using polynomials
        minerals = [["Cpx"],["Ol_spl"],["Spl"],["Ol_gnt"],["Gnt"]] 
 
        # Make Dictionaries for Polynomial Constants
        # Table 1 in Main Text
        a = {}
        a["Cpx"] = np.array([0.037, -0.229, -0.606])
        a["Ol_spl"] = np.array([-0.115, 0.314, 0.318])
        a["Ol_gnt"] = np.array([0.048, -0.558, 1.298])
        a["Spl"] = np.array([0.026, -0.013, -0.087])
        a["Gnt"] = np.array([-0.005, 0.078, -0.557])

        b = {}
        b["Cpx"] = np.array([-0.011, 0.112, 0.058])
        b["Ol_spl"] = np.array([-0.039, 0.126, 0.419])
        b["Ol_gnt"] = np.array([-0.003, 0.035, 0.445])
        b["Spl"] = np.array([-0.004, 0.004, 0.02])
        b["Gnt"] = np.array([-0.001, 0.033, 0.008])

        # Make dictionay of mineral proportions as a function of melt fraction.
        MinComp = {}
        
        for mineral in minerals:
            
            MinComp[mineral[0]] = np.zeros_like(X)
            
            # EQs 10 and 11 in Main Text.
            for i, Xi in enumerate (X):
                MinComp[mineral[0]][i] = (a[mineral[0]][1]*P[i] + a[mineral[0]][0]*(P[i]**2.) + a[mineral[0]][2])*X[i] + (b[mineral[0]][1]*P[i] + b[mineral[0]][0]*(P[i]**2.) + b[mineral[0]][2])    
            
        # Accounting for Mantle Depletion
        # Depleted mantle assumed to have 4% less cpx and 4% more ol than primitive mantle.
        # Linear variation on cpx and ol content as a function of eNd between 0 and 10. 
        MinComp["Cpx"] = MinComp["Cpx"] - (0.04 * (eNd/10.))
        MinComp["Ol_spl"] = MinComp["Ol_spl"] + (0.04 * (eNd/10.))
        MinComp["Ol_gnt"] = MinComp["Ol_gnt"] + (0.04 * (eNd/10.))
        
        # Ensure no negative mineral proportions 
        for mineral in minerals:
            for i, Xi in enumerate (X):    
                if MinComp[mineral[0]][i] < 0.:
                    MinComp[mineral[0]][i] = 0.
        
        # Opx makes up whats left of the modal mineralogy
        # EQ 13 in Main Text.
        MinComp["Opx_spl"] = np.zeros_like(X)
        MinComp["Opx_gnt"] = np.zeros_like(X)
        for i, Xi in enumerate (X):
            MinComp["Opx_spl"][i] = 1. - X[i] - MinComp["Ol_spl"][i] - MinComp["Cpx"][i] - MinComp["Spl"][i] 
            MinComp["Opx_gnt"][i] = 1. - X[i] - MinComp["Ol_gnt"][i] - MinComp["Cpx"][i] - MinComp["Gnt"][i] 
        
        # Make dictionaries of mineral proportions including both spl and gnt peridotite.
        Fn_spl = {}
        Fn_gnt = {}
        
        for mineral in [["Ol_spl","0"],["Opx_spl","1"],["Cpx","2"],["Spl","4"]]:
            Fn_spl[mineral[1]] = np.zeros_like(X)
            for i, Xi in enumerate (X):
                Fn_spl[mineral[1]][i] = MinComp[mineral[0]][i]

        for mineral in [["Ol_gnt","0"],["Opx_gnt","1"],["Cpx","2"],["Gnt","5"]]:
            Fn_gnt[mineral[1]] = np.zeros_like(X)
            for i, Xi in enumerate (X):
                Fn_gnt[mineral[1]][i] = MinComp[mineral[0]][i]
                
        # Minerals not included in each peridotite composition added as zeros.
        Fn_spl["3"] = np.zeros_like(X)
        Fn_spl["5"] = np.zeros_like(X)
        Fn_gnt["3"] = np.zeros_like(X)
        Fn_gnt["4"] = np.zeros_like(X)                
        
        # Make dictionary of melt stoiceometry as a function of melt fraction.
        pn_spl = {}
        pn_gnt = {}
        
        for j in range(0,6,1):
            pn_spl[str(j)] = np.zeros_like(X)
            pn_gnt[str(j)] = np.zeros_like(X)
            for i in range(1,(int(len(X))-1),1):
                pn_spl[str(j)][i] = (Fn_spl[str(j)][i-1] - Fn_spl[str(j)][i+1]) / (X_inc * 2.)
                pn_gnt[str(j)][i] = (Fn_gnt[str(j)][i-1] - Fn_gnt[str(j)][i+1]) / (X_inc * 2.)
            
            pn_spl[str(j)][0] = pn_spl[str(j)][1]
            pn_gnt[str(j)][0] = pn_gnt[str(j)][1]
            
        return Fn_spl, Fn_gnt, pn_spl, pn_gnt
    
#-------------
    
    def calc_depth_mineralogy(self, X, P, X_gnt_out, X_spl_in, eNd):
        """
        Calculates changing mineralogy as a function of depth
        taking into account the spinel-garnet-transition zone.
        
        Combine spinel and garnet mineralogy and melt stoicheometry linearly along
        the spinel garnet transition zone.
        """
        
        Fn_spl, Fn_gnt, pn_spl, pn_gnt = self.calc_polynomials(X, P, eNd)

        Fn = {}
        pn = {}
        
        for j in range(0,6,1):
            Fn[str(j)] = np.zeros_like(X)
            pn[str(j)] = np.zeros_like(X)

            for i, Xi in enumerate(X):
        
                if X[i] < X_spl_in:
                    Fn[str(j)][i] = Fn_gnt[str(j)][i]
                    pn[str(j)][i] = pn_gnt[str(j)][i]                    
                
                elif ((X[i] >= X_spl_in) and (X[i] < X_gnt_out)):
                    Fn[str(j)][i] = ((Fn_gnt[str(j)][i] * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (Fn_spl[str(j)][i] * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                    pn[str(j)][i] = ((pn_gnt[str(j)][i] * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (pn_spl[str(j)][i] * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                
                elif X[i] >= X_gnt_out:
                    Fn[str(j)][i] = Fn_spl[str(j)][i]
                    pn[str(j)][i] = pn_spl[str(j)][i]
                
        return Fn, pn
    
#-------------  

    def calc_Brice1975(self, Do, E, ro, ri, T):
        """
        Lattice strain equation (Brice, 1975, Wood and Blundy, 1997).
        EQ 4 in Supplementary Material
        """

        D_n = Do * np.exp(-4. * np.pi * const.Avagadro * E * ((0.5 * ro * (ro - ri)**2.) - ((ro - ri)**3.)/3.) / (const.R * T))

        return (D_n)

#-------------

    def calc_Dn(self, P, T, X, ri, D, val, X_spl_in, X_gnt_out):
        """
        Calculates D_bar and P_bar (bulk partition in solid and melt) as a function of depth.
        EQ 8 in Main Text.
        Assumes D in all phases is depth dependent.
        Cpx, opx, gnt, ol +3 calculations from Sun and Liang, 2012/2013, Min. Pet. and Yao et al., 2012.
        Other valencies determined using methods in Wood and Blundy, 2014.
        j 1-5 = ol, opx, cpx, plg, spl, gnt
        """

        Dn = {}
        for j in range(0,6,1):
            Dn[str(j)] = np.zeros_like(X)

        for i, Xi in enumerate(X):
        
            for j in range(0,6,1):

                # Spinel
                # Constant D for SPL from McKenzie and O'Nions (1995)
                if (j == 4):
                    Dn[str(j)][i] = D[j]
 
                #------------------
    
                # Olivine
                elif (j == 0):

                    # Sun and Liang, 2013
                    if val == 3.:

                        # Values for chemical constants calculated using EQ 50
                        # and Figure 3 in Supplementary Material.
                        F_Al_gnt = 0.00564
                        Mg_num_gnt = (0.07 * X[i]) + 0.897
                        F_Al_spl = 0.00156
                        Mg_num_spl = (0.059 * X[i]) + 0.904

                        if X[i] < X_spl_in:
                            F_Al = F_Al_gnt
                            Mg_num = F_Al_gnt
                    
                        elif ((X[i] >= X_spl_in) and (X[i] < X_gnt_out)):
                            F_Al = ((F_Al_gnt * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (F_Al_spl * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                            Mg_num = ((Mg_num_gnt * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (Mg_num_spl * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                    
                        elif X[i] >= X_gnt_out:
                            F_Al = F_Al_spl
                            Mg_num = Mg_num_spl

                        # Constants for lattice strain equation calculated
                        # using parameterization of Sun and Liang (2013)
                        # and listed in EQ 5-7 in Supplementary Material.
                        Do = np.exp(- 0.45 - (0.11 * P[i]) + (1.54 * F_Al) - ((1.94 * 10.**-2.) * Mg_num))
                        ro = 0.72
                        E = 426.  
                        
                        # Caluclate D using lattice strain EQ (EQ 4 in Supplementary Materials).
                        Dn[str(j)][i] = self.calc_Brice1975( Do, E * (10.**9.), ro * (10.**-10.), ri[0], T[i])
                   
                    # If Valancy is not 3+ then use constant value for D
                    # from McKenzie and O'Nions, 1995.
                    else:
                        Dn[str(j)][i] = D[j]

                #------------------
                
                # Orthopyroxene
                elif (j == 1):

                    # Values for chemical constants calculated using EQ 51-53
                    # and Figure 4 in Supplementary Material.
                    X_Ca_M2 = (-0.756 * X[i]**2.) + (0.273 * X[i]) + 0.063
                    X_Mg_M2 = (0.692 * X[i]**2.) + (-0.176 * X[i]) + 0.834
                    X_Al_T = (-0.675 * X[i]**2.) + (0.041 * X[i]) + 0.146
                    
                    # Constants for lattice strain equation calculated
                    # using parameterization of Yao et al (2012)
                    # and listed in EQ 10-12 in Supplementary Material.
                    a = - 5.37 + (38700./(const.R*T[i]))
                    Do = np.exp(a + (3.54 * X_Al_T) + (3.56 * X_Ca_M2))
                    ro = 0.69 + (0.23 * X_Mg_M2) + (0.43 * X_Ca_M2)
                    E = (- 1.37 + (1.85 * ro) - (0.53 * X_Ca_M2)) * 10.**3.
                    
                    # Wood and Blundy, 2014, Treatise parameterization as described by
                    # EQ 17-20 in Supplementary Materials.
                    if val == 2.:
                        E_v2 = (2./3.) * E * 10.**9.
                        ro_v2 = (ro + 0.08) * 10.**-10.
                        ri_Mg = 0.72 * 10.**-10.
                        
                        # Calculate D using lattice strain EQ (EQ 4 in Supplementary Materials).
                        Dn[str(j)][i] = np.exp(-4.*np.pi*const.Avagadro*E_v2*((ri_Mg/2.)*(ri_Mg**2. - ri[1]**2.)+(1./3.)*(ri[1]**3. - ri_Mg**3.))/(const.R*T[i]))

                    # If valency = 3+ calculate D using lattice strain EQ (EQ 4 in Supplementary Materials).
                    elif val == 3.:
                        Dn[str(j)][i] = self.calc_Brice1975( Do, E * (10.**9.), ro * (10.**-10.), ri[1], T[i])
                    
                    # If Valancy is not 3+ or 2+ then use constant value for D
                    # from McKenzie and O'Nions, 1995.                    
                    else:
                        Dn[str(j)][i] = D[j]

                #------------------

                # Clinopyroxene.
                elif (j == 2):
                    
                    # Values for chemical constants calculated using EQ 54-57
                    # and Figure 5 in Supplementary Material.                    
                    X_Mg_M2_spl = (0.583 * X[i]) + 0.223
                    X_Al_T_spl = (-0.177 * X[i]) + 0.154
                    X_Al_M1_spl = (-0.438 * X[i]) + 0.137
                    X_Mg_M1_spl = (0.425 * X[i]) + 0.741
                    X_Mg_Mel_spl = (0.191 * X[i]) + 0.793
                    X_Mg_M2_gnt = (0.422 * X[i]) + 0.547
                    X_Al_T_gnt = (-0.013 * X[i]) + 0.061
                    X_Al_M1_gnt = (-0.114 * X[i]) + 0.099
                    X_Mg_M1_gnt = (0.14 * X[i]) + 0.722
                    X_Mg_Mel_gnt = (0.207 * X[i]) + 0.701                    
                    
                    # Clinopyroxene changes chemical composition as a function of X
                    # Differently in spl and gnt stability fields.
                    if X[i] < X_spl_in:
                        X_Mg_M2 = X_Mg_M2_gnt
                        X_Al_T = X_Al_T_gnt
                        X_Al_M1 = X_Al_M1_gnt
                        X_Mg_M1 = X_Mg_M1_gnt
                        X_Mg_Mel = X_Mg_Mel_gnt

                    elif ((X[i] >= X_spl_in) and (X[i] < X_gnt_out)):
                        X_Mg_M2 = ((X_Mg_M2_gnt * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (X_Mg_M2_spl * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                        X_Al_T = ((X_Al_T_gnt * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (X_Al_T_spl * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                        X_Al_M1 = ((X_Al_M1_gnt * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (X_Al_M1_spl * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                        X_Mg_M1 = ((X_Mg_M1_gnt * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (X_Mg_M1_spl * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))
                        X_Mg_Mel = ((X_Mg_Mel_gnt * (X_gnt_out - X[i])/(X_gnt_out - X_spl_in)) + (X_Mg_Mel_spl * (X[i] - X_spl_in)/(X_gnt_out - X_spl_in)))                        

                    elif X[i] > X_gnt_out:
                        X_Mg_M2 = X_Mg_M2_spl
                        X_Al_T = X_Al_T_spl
                        X_Al_M1 = X_Al_M1_spl
                        X_Mg_M1 = X_Mg_M1_spl
                        X_Mg_Mel = X_Mg_Mel_spl

                    # Water content set as zero
                    X_H2O = 0.

                    # Constants for lattice strain equation calculated
                    # using parameterization of Sun and Liang (2012)
                    # and listed in EQ 21-23 in Supplementary Material.
                    a = ((7.19 * 10.**4.) / (const.R * T[i])) - 7.14
                    Do = np.exp(a + (4.37 * X_Al_T) + (1.98 * X_Mg_M2) - (0.91 * X_H2O))
                    ro = 1.066 - (0.104 * X_Al_M1) - (0.212 * X_Mg_M2)
                    E = ((2.27 * ro) - 2.) * 10.**3.
                    
                    if val == 1.:
                        # Use parameterization of Blundy et al., 1995 for Na
                        # EQ 24-25 in Supplementary Materials.
                        D_Na = np.exp(((10367. + (2100.*P[i]) - (165.*P[i]**2.))/T[i]) - 10.27 + (0.358*P[i]) - (0.0184*P[i]**2.))
                        ri_Na = 1.18 * 10.**-10.
                        
                        # Use Hazen-Finger relationships for E and approximation for ro
                        # from calculation for 3+ valency (Wood and Blundy, 2014, Treatise)
                        # EQ 26-28 in Supplementary Materials
                        E_v1 = (1./3.) * E * 10.**9.
                        ro_v1 = (ro + 0.12) * 10.**-10.
                        # All +1 ions calculated with respect to Na
                        b = (ro_v1/2.)*(ri_Na**2. - ri[1]**2.) + (1./3.)*(ri[1]**3. - ri_Na**3.)
                        Dn[str(j)][i] = D_Na * np.exp((-4. * np.pi * const.Avagadro * E_v1 * b)/(const.R * T[i]))
                    
                    elif val == 2.:
                        # For 2+ cations use parameterization as listed in Wood and Blundy, (2014).
                        # EQ 29-31
                        # Do from Blundy and Wood, 2003, EPSL.
                        Do_v2 = 4.25
                        # Use Hazen-Finger relationships for E
                        E_v2 = (2./3.) * E * 10.**9.
                        # ro from (Wood and Blundy, 2014, Treatise)
                        ro_v2 = (ro + 0.06) * 10.**-10.
                        Dn[str(j)][i] = self.calc_Brice1975( Do_v2, E_v2, ro_v2, ri[1], T[i])
                        
                    # If valency = 3+ calculate D using lattice strain EQ (EQ 4 in Supplementary Materials).
                    elif val == 3.:
                        
                        Dn[str(j)][i] = self.calc_Brice1975( Do, E * (10.**9.), ro * (10.**-10.), ri[1], T[i])
                   
                    # Wood and Blundy, 2014, Treatise parameterization as described by
                    # EQ 32-38 in Supplementary Materials.
                    elif val == 4.:
                        # Use Hazen-Finger relationships for E
                        E_v4 = (4./3.) * E * 10.**9.
                        # ro from (Wood and Blundy, 2014, Treatise)
                        ro_v4 = ro * 10.**-10.
                        # Using parameterisation of Landwehr et al., 2001 for Th
                        ri_Th = 1.035 * 10.**-10.
                        Y_Mg_M1 = np.exp(((1. - X_Mg_M1)**2.) * 902. / T[i])
                        Y_Th_M2 = np.exp(((ro_v4/2.)*(ri_Th - ro_v4)**2. + (1./3.)*(ri_Th - ro_v4)**3.)*(4*np.pi*const.Avagadro*E_v4)/(const.R*T[i])) 
                        D_Th = np.exp((214790. - (175.5*T[i]) + (16420.*P[i]) - (1500.*P[i]**2.))/const.R*T[i]) * X_Mg_Mel / (X_Mg_M1 * Y_Mg_M1 * Y_Th_M2)
                        b = (ro_v4/2.)*(ri_Th**2. - ri[1]**2.) + (1./3.)*(ri[1]**3. - ri_Th**3.)
                        Dn[str(j)][i] = D_Th*np.exp((-910.17*E_v4*b)/T[i])
                    
                    # Hard to parameterize, using constant values instead (McKenzie and O'Nions, 1995).
                    elif val == 5.:
                        Dn[str(j)][i] = D[j]

                #------------------                    

                # D for Plagioclase (not used in current formulation).
                # Constant values from McKenzie and O'Nions (1995).
                elif (j == 3):
                
                    Dn[str(j)][i] = D[j]

                #------------------

                # D for Garnet
                elif (j == 5):

                    # Value for F_Ca calculated using EQ 58
                    # and Figure 6 in Supplementary Material.     
                    F_Ca = (-0.247 * X[i]) + 0.355
                    
                    # Constants for lattice strain equation calculated
                    # using parameterization of Sun and Liang (2013)
                    # and listed in EQ 39-41 in Supplementary Material.
                    Do = np.exp(-2.05 + ((91700. - (3471.3*P[i]) + (91.35*P[i]**2.))/(const.R*T[i])) - 1.02 * F_Ca)
                    ro = 0.78 + (0.155 * F_Ca)
                    E = (-1.62 + (2.29 * ro)) * 10.**3.                 

                    # If valency = 3+ calculate D using lattice strain EQ (EQ 4 in Supplementary Materials).  
                    if val == 3.:
                        Dn[str(j)][i] = self.calc_Brice1975(Do, E * 10.**9., ro * 10.**-10., ri[1], T[i])
                    
                    # For 2+ cations constants for lattice strain equation calculated
                    # using parameterization from Wood and Blundy, 2014
                    # relative to the constants for 3+ cations.
                    elif val == 2.:
                        ri_Mg = 0.72 * 10.**-10.
                        F_Ca = (-0.247 * X[i]) + 0.355
                        D_Mg = np.exp((258210. - (141.5*T[i]) + (5418.*P[i]))/(3*const.R*T[i])) / np.exp((19000. * (F_Ca**2.))/(const.R*T[i]))
                        ro_v2 = (0.053 + ro) * 10.**-10.
                        E_v2 = (2./3.) * E * 10.**9.
                        b = (ro_v2/2.)*(ri_Mg**2. - ri[1]**2.) + (1./3.)*(ri[1]**3. - ri_Mg**3.)
                        Dn[str(j)][i] = D_Mg * np.exp(-4. * np.pi * const.Avagadro * E_v2 * b / (const.R * T[i]))        

                    # For 4+ cations constants for lattice strain equation calculated
                    # using parameterization of Mallmann and O'Neill, 2007
                    # EQ 47-49 in Supplementary Materials.
                    elif val == 4.:
                        Do_v4 = 4.38
                        E_v4 = 2753 * 10.**9.
                        ro_v4 = 0.6626 * 10.**-10.
                        Dn[str(j)][i] = self.calc_Brice1975(Do_v4, E_v4, ro_v4, ri[1], T[i])

                    # valency of +1 and +5 too poorly parameterized
                    # and so constant values are used from McKenzie and O'Nions, (1995).
                    else:
                        Dn[str(j)][i] = D[j]

        return Dn

#-------------

    def calc_D_Bar_P_Bar(self, X, Dn, Fn, pn):
        """
        Calculates D_bar and P_bar (bulk partition in solid and melt) as a function of depth.
        j 1-5 = ol, opx, cpx, plg, spl, gnt
        EQ 8 in Main Text.
        """
        
        D_bar = np.zeros_like(X)
        P_bar = np.zeros_like(X)
        
        for i, Xi in enumerate(X):
            D_bar[i] = (Dn["0"][i]*Fn["0"][i]) + (Dn["1"][i]*Fn["1"][i]) + (Dn["2"][i]*Fn["2"][i]) + (Dn["3"][i]*Fn["3"][i]) + (Dn["4"][i]*Fn["4"][i]) + (Dn["5"][i]*Fn["5"][i])
                
            P_bar[i] = (Dn["0"][i]*pn["0"][i]) + (Dn["1"][i]*pn["1"][i]) + (Dn["2"][i]*pn["2"][i]) + (Dn["3"][i]*pn["3"][i]) + (Dn["4"][i]*pn["4"][i]) + (Dn["5"][i]*pn["5"][i])

        return D_bar, P_bar

#-------------

    def calc_dcs_dX(self, X, cs, D_bar, P_bar):
        """
        Combination of EQ 3 + 4 from White et al., 1992.
        Must be solved using 4th order Runge Kutta scheme.
        EQ 7 in Main Text.
        """

        dcs_dX = - ((cs * (1. - X)/(D_bar - (P_bar * X))) - cs)/(1. - X)
        if dcs_dX > cs:
            dcs_dX = cs
            
        return (dcs_dX)

#-------------

    def calc_melt_comp(self, cs_0, X, D_bar, P_bar):
        """
        Calculates melt composition at each step along the meltpath.
     
        """
        
        X_inc = self.X_inc
        X_0 = self.X_0

        # Instantaneous element concentration in solid
        cs = np.zeros_like(X)
        # Initial concentration in the solid at X = 0.        
        cs[0] = cs_0

        # Instantaneous element concentration in liquid 
        cl = np.zeros_like(X)
        # Initial concentration of element in melt at X = 0.
        cl_0 = (cs_0 * (1 - X_0)/(D_bar[0] - (P_bar[0] * X_0)))
        cl[0] = cl_0

        # Use a fourth-order Runge Kutta scheme to evaluate value of cs and cl
        # (i.e. the concentration of each element in the solid and melt, repsectively)
        # at position X, given the values of cs_0 and cl_0 at previous position X_0.
        for i, Xi in enumerate(X[:-1]):        

            if (X[i] == 0.):
                cl[i] = 0.
                cs[i+1] = cs_0
            else:
                # Once concentration of element in the solid is exhausted cs and cl = 0.
                if (cs[i] > 0.0001):
                    k1 = 1. * self.calc_dcs_dX( X[i],             cs[i],                D_bar[i], P_bar[i] )
                    k2 = 1. * self.calc_dcs_dX( X[i] + 0.5*X_inc, cs[i] + X_inc*0.5*k1, ((D_bar[i]+D_bar[i+1]) / 2.), ((P_bar[i]+P_bar[i+1]) / 2.) )
                    k3 = 1. * self.calc_dcs_dX( X[i] + 0.5*X_inc, cs[i] + X_inc*0.5*k2, ((D_bar[i]+D_bar[i+1]) / 2.), ((P_bar[i]+P_bar[i+1]) / 2.) )
                    k4 = 1. * self.calc_dcs_dX( X[i] + X_inc,     cs[i] + X_inc*k3,     D_bar[i+1], P_bar[i+1] )
                    cs[i+1] = (cs[i] + (X_inc/6.) * (k1 + 2.*k2 + 2.*k3 + k4))
                    cl[i] = cs[i] * (1. - X[i])/(D_bar[i] - (P_bar[i] * X[i]))
                    # Forces element to stay in the melt phase
                    if (cl[i] < 0.):
                        cl[i] = 0.
                        cs[i+1] = cs[i] / (1. - X_inc)
                else:
                    cl[i] = 0.
                    cs[i] = 0.

        return cl
    
#-------------

    def calc_melt_mineralogy(self, X, P, X_km, h_km, h_dash, eNd):
        """
        Calculates mineralogy and melt stoichiometry as a function of depth.        
        """

        # Determine Melt Fractions Corresponding to Mantle Phase Transitions.
        X_spl_out, X_plg_in, X_gnt_out, X_spl_in = self.phase_X(X_km, h_km, h_dash)
        # Calculate Depth Dependent Mineralogy and Melt Stoichiometry.
        Fn, pn = self.calc_depth_mineralogy(X, P, X_gnt_out, X_spl_in, eNd)

        return Fn, pn, X_gnt_out, X_spl_in  

#-------------
# Final Routine to Calculate Melt Composition for a Given Incompatible Element Along a Melt Path
#-------------    

    def calc_particle_comp(self, Tp, eNd, Conc1, Conc2, Part, VAL, RI, elements, party, parttemp, partKatzF):

        # Make dictionary of elemental instantaneous concentrations for each element (cl)
        # outputted for each timestep in the input model (el_cl).
        el_cl = {}

        # Take pressure, temperature, melt fraction paths from model (party, parttemp, partKatzF)
        # which are outputted at each time step and generate paths at Constant Intervals of melt fraction.
        # P = pressure, T = temperature, X = melt fraction.
        P, T, X = self.calc_PTX(Tp, party, parttemp, partKatzF)
            
        # Convert Melt Path from constant intervals of X to constant intervals of depth.
        # These are used to calculate melt fractions which correspond to transition zones
        # which are defined in depth space.
        # h = depth in intervals of X_inc, h_km = depth in 1 km intervals, h_dash = , X_km = X in 1 km depth intervals.
        h, h_km, h_dash, X_km = self.calc_PTX_h(P, T, X)
        
        # If the length of the line is <2 then it cannot be integrated to calculate el_cl
        # and so el_cl = nan.
        if len(h_km) < 2:

            for i in elements:

                el_cl[str(i)] = np.full((len(partKatzF),1), np.nan)

        # If the length of the line is >2 then calculate el_cl
        else:
 
            # Determine Melt Fractions Corresponding to Mantle Phase Transitions (X_gnt_out, X_spl_in).
            # Calculate modal mineralogy (Fn), and melt stoichiometry (pn) as a function of X.
            h_dash = int(h_dash * -1.)
            Fn, pn, X_gnt_out, X_spl_in = self.calc_melt_mineralogy(X, P, X_km, h_km, h_dash, eNd)
           
            #Calculate Incompatible Element Composition for each element in elements.   
            for i in elements:
   
                # Read in constant partition coefficients (D), element radii (ri) and valency (val).
                D = Part[str(i)]
                ri = RI[str(i)] * 10.**-10.
                val = VAL[str(i)]
       
                # If mixed source linearly mix primitive (Conc1) and depleted (Conc2) compositions 
                # according to eNd value to form original source composition (cs_0).
                cs_0 = (((10 - eNd)/10.) * Conc1[str(i)]) + ((eNd/10.) * Conc2[str(i)])
       
                # Calculate partition coefficients for each mineral as function of PTX (Dn).
                Dn = self.calc_Dn(P, T, X, ri, D, val, X_spl_in, X_gnt_out)
                
                # Calculate bulk partition coefficient in the solid (D_bar) and melt (P_bar).
                D_bar, P_bar = self.calc_D_Bar_P_Bar(X, Dn, Fn, pn)
                
                # Calculate cl 
                cl= self.calc_melt_comp(cs_0, X, D_bar, P_bar)
                
                # Interpolate cl at each time step
                el_cl[str(i)] = np.interp(partKatzF, X, cl)
                
                # Remove first line
                el_cl[str(i)][0] = np.nan
                
                # First time step gives strange result due to only one value to integrate, make first value equal second value.
                if len(el_cl[str(i)]) > 3:
                    el_cl[str(i)][1] = el_cl[str(i)][2]
 
        return el_cl
