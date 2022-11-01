import pandas as pd 
import numpy as np
import lmfit as lm
from lmfit import Parameters, Minimizer
import plotly.express as px
from plotly.subplots import make_subplots

# Richards Model
@np.vectorize
def richards(L,a,b,t,tj):
    j=L-L*(1+a*np.exp(b*(t-tj)))**(-1/a)
    return j


def obj_fun(params, t_ini, t_fin, df):
    '''Objective function'''
    window = t_fin - t_ini, 
    pars = params.valuesdict()
    L = pars['L1']
    tp = pars['tp1']
    a = pars['a1']
    b = pars['b1']
    
    t_range = np.arange(t_fin - t_ini)
    richfun = richards(L,a,b,t_range,tp) 
    serie = df.loc[t_ini:t_fin].casos_cum.values
    
    mse = (serie-richfun)**2/window
                     
    return mse

def get_SIR_pars(rp: dict):
    """
    Returns the SIR parameters based on the Richards model's parameters (rp) 
    """
    a = rp['a1']
    b= rp['b1']
    tc = rp['tp1']
    pars = {
        'beta': b/a,
        'gamma': (b/a)-b,
        'R0': (b/a)/((b/a)-b),
        'tc': tc
    }
    return pars

 
def otim(df, t_ini, t_fin, verbose=False):
    df.reset_index(inplace=True)
    df['casos_cum'] = df.casos.cumsum()
    params = Parameters()
    params.add('gamma', min = 0.3, max = 0.33)
    params.add('L1', min = 1., max = 1e5)
    params.add('tp1', min = 10, max = 30)
    params.add('b1', min = 1e-6, max = 1)
    params.add('a1', expr='b1/(gamma + b1)', min = 0.001, max = 1)
    
    window = min(int(t_fin - t_ini), len(df))
    t_range = np.arange(window)
 
    out = lm.minimize(obj_fun, params, args = (0, window, df), method = 'diferential_evolution')
    if verbose:
        if out.success:
            print(f"found  match after {out.nfev} tries")
        else:
            print(f"No match found")
            return False, df

    pars = out.params
    pars = pars.valuesdict()

    # serie = df.loc[t_ini:t_fin].casos_cum.values
    richfun_opt = richards(pars['L1'],pars['a1'],pars['b1'],t_range,pars['tp1'])
    
    df = df.iloc[:window]
    
    df['richards'] =  richfun_opt + np.zeros(window) 
    
    
    return  out, df

class EpiScanner:
    def __init__(self, last_week: int, data: pd.DataFrame):
        """
        Scans a time series for an epidemic curve
        :Parameters:
        last_week: Last week of epidemic season 
        data: dataframe with the series from all cities
        """
        self.window = last_week
        self.data = data
        self.results = defaultdict(list)
        self.curves = defaultdict(list)
        
    def _filter_city(self, geocode):
        dfcity = self.data[self.data.municipio_geocodigo==geocode]
        dfcity.sort_index(inplace = True)
        dfcity['casos_cum'] = dfcity.casos.cumsum()
        return dfcity
    
    def scan(self, geocode, verbose=True):
        df = self._filter_city(geocode)
        df['year'] = [i.year for i in df.index]
        for y in set(df.year.values):
            if verbose:
                print(f"Scanning year {y}")
            dfy = df[df.year==y]
            has_transmission = dfy.transmissao.sum() > 3
            if not has_transmission:
                if verbose:
                    print(f"There where less that 3 weeks with Rt>1 in {geocode} in {y}.\nSkipping analysis")
                continue
            out, curve = otim(dfy[['casos','casos_cum']].iloc[0:self.window],0,self.window)
            self._save_results(geocode, y, out, curve)
            if out.success:
                if verbose:
                    print(f"R0 in {y}: {self.results[geocode][-1]['sir_pars']['R0']}")
            
    def _save_results(self, geocode, year, results, curve):
        self.results[geocode].append({
            'year': year,
            'success': results.success,
            'params': results.params.valuesdict(),
            'sir_pars': get_SIR_pars(results.params.valuesdict())
        })
        self.curves[geocode].append({
                'year': year,
                'df': curve
                })
        
    def plot_fit(self, geocode, year=0):
        if year == 0:
            nyears = len(self.curves[geocode])
            nrows = nyears//2 if nyears % 2 == 0 else nyears//2+1
            ncols = 2
        else:
            ncols = 1
            nrows = 1
        fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize= (10,10))
        axes = axes.ravel()
        i=0
        for curve in self.curves[geocode]:
            y = curve['year']
            df = curve['df']
            df.set_index('data_iniSE', inplace=True)
            if year != 0 and y != year:
                continue
            df.casos_cum.plot.area(ax=axes[i], alpha=0.3, color='r', label=f'data_{y}',rot=45)
            df.richards.plot(ax=axes[i],label='model',use_index=True)
            axes[i].legend()
            i += 1
            
    def to_csv(self, fname):
        data = {'geocode':[],'year':[],'peak_week':[],'beta':[],'gamma':[],'R0':[],'total_cases':[],'alpha':[]}
        i = 0 
        for gc, curve in self.curves.items():
            for c in curve:
                data['geocode'].append(gc)
                data['year'].append(c['year'])
                params = [p['params'] for p in  self.results[gc] if p['year']==c['year']][0]
                sir_params = [p['sir_pars'] for p in  self.results[gc] if p['year']==c['year']][0]
                data['peak_week'].append(params['tp1'])
                data['total_cases'].append(params['L1'])
                data['alpha'].append(params['a1'])
                data['beta'].append(sir_params['beta'])
                data['gamma'].append(sir_params['gamma'])
                data['R0'].append(sir_params['R0'])
        dfpars = pd.DataFrame(data)
        dfpars = dfpars.to_csv(f'{fname}.csv.gz')
