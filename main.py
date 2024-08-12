import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from itertools import combinations
from termcolor import colored
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat
from statannotations.Annotator import Annotator
import time
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import itertools

# seaborn==0.13.2
# statsannotations==0.6.0

mpl.rcParams['font.family'] = 'Helvetica Neue'

main_folders = []
main_folders.append(os.path.abspath('./Experiments/2024-04-04_TraT_conj_assay/'))
main_folders.append(os.path.abspath('./Experiments/2024-04-11_TraT_conj_assay/'))
main_folders.append(os.path.abspath('./Experiments/2024-04-17_TraT_conjugation/'))

bacseg_folders = ['20240710_bacseg_new']

master_order = ('Empty', 'T1', 'T2', 'T3', 'E', '199', 'WT', '193', '99', '253', '100', '255', '101', '256', '102', '257') #order to plot the samples on the x axis

show_plots = True
stripplot_points_color = (0.2, 0.2, 0.2)
stripplot_points_opacity = 0.8
alpha = 0.05
hide_ns = False
compute_stats = True
show_p_values = False
show_QQ_plot = False
save_plot_name = 0

durations = pd.DataFrame(columns=['point', 'duration'])
start = time.time()

results = pd.DataFrame({'sample': [], 'total_donors': [], 'total_recipients': [], 'total_transconjugants': [],
                        'conjugation_efficiency': [],
                        'transconjugants_per_donor': [],
                        'normalized_conjugation_efficiency': [],
                        'transconjugants_per_FOV': [],
                        'donor_to_recipient_ratio': []})

dfs = []
day = 1
for main_folder in main_folders :
    for folder in [folder for folder in os.listdir(main_folder) if os.path.isdir(main_folder + '/' + folder)]:
        for bacseg_folder in bacseg_folders :
            if os.path.isdir(f'{main_folder}/{folder}/{bacseg_folder}/'):
                if 'ignore' in f'{main_folder}/{folder}/{bacseg_folder}/' :
                    continue
                new_df = pd.read_csv(f'{main_folder}/{folder}/{bacseg_folder}/Conjugation_results.csv')
                new_df.insert(0, 'sample', folder)
                new_df['day'] = day
                dfs.append(new_df)
                break
    day += 1
results = pd.concat(dfs, ignore_index=True)

results['ratio'] = results['total_donors'] / (results['total_recipients'] + results['total_transconjugants'])
results['total_cells'] = results['total_donors'] + results['total_recipients'] + results['total_transconjugants']

order = [e for e in master_order if e in results['sample'].values]

# x-axis labels
labels = {
    'Empty': 'Vector',
    'E': 'Vector',
    '199': 'Empty',
    'WT': 'WT',
    '193': 'TraT-WT',
    'T1': 'TraT$_{NSAGA}^{WT}$',
    'T2': 'TraT$_{NSAGG}$',
    'T3': 'TraT$_{SSAGA}$',
    '99': 'D170A',
    '253': 'D170A',
    '100': 'R213A',
    '255': 'R213A',
    '101': 'R213E',
    '256': 'R213E',
    '102': 'D170A\nR213A',
    '257': 'D170A\nR213A'
}

def shapiro(l, alpha=alpha):
    for d in l:
        s, p = stats.shapiro(d)
        if p < alpha:
            return False, p
    return True, p

def bartlett(l, alpha=alpha):
    s, p = stats.bartlett(*l)
    if p < alpha:
        return False, p
    return True, p

def levene(l, alpha=alpha, center='median'):
    s, p = stats.levene(*l, center=center)
    if p < alpha:
        return False, p
    return True, p

def f_oneway(l, alpha=alpha):
    s, p = stats.f_oneway(*l)
    if p < alpha:
        return False, p
    return True, p

def tukey(l):
    res = stats.tukey_hsd(*l)
    result = []
    for i, j in combinations(range(len(l)), 2):
        res_p = res.pvalue[i, j]
        result.append(res_p)
        if res_p < alpha:
            print(f"   {order[i]}-{order[j]} : {colored('DIFF', 'green')}, p={res_p}")
        else:
            print(f"   {order[i]}-{order[j]} : {'NOT DIFF'}, p={res_p}")
    return result

def kruskal(l, alpha=alpha):
    s, p = stats.kruskal(*l)
    if p < alpha:
        return False, p
    return True, p

def wilcoxon(l, alpha=alpha):
    print('PAIRWISE WILCOXON-MANN-WHITNEY COMP :')
    result = []
    for v in combinations(enumerate(l), 2):
        (i, e), (j, f) = v
        res_v, res_p = stats.mannwhitneyu(e, f)
        result.append(res_p)
        if res_p < alpha:
            print(f"   {order[i]}-{order[j]} : {colored('DIFF', 'green')}, p={res_p}")
        else:
            print(f"   {order[i]}-{order[j]} : {'NOT DIFF'}, p={res_p}")
    return result

def perm(test, l, nperm=999):
    # Calculate the observed Bartlett test statistic
    K_ref = test(*l).statistic

    # Initialize an array to store permuted test statistics
    K_perm = np.zeros(nperm + 1)
    K_perm[0] = K_ref

    # Perform permutations and calculate permuted test statistics
    dfs = []
    for i, data in enumerate(l):
        new_df = pd.DataFrame(np.array(data), columns=['data'])
        new_df.insert(0, 'sample', i)
        dfs.append(new_df)
    all_data = pd.concat(dfs, ignore_index=False)

    for j in range(1, nperm + 1, 1):
        perm_fact = np.random.permutation(all_data['sample'])
        all_data['sample'] = perm_fact
        K_perm[j] = test(*[all_data[all_data['sample'] == sample]['data'] for sample in list(range(len(l)))]).statistic

    # Calculate p-value
    pvalue = np.sum((K_perm + np.finfo(float).eps / 2) >= K_ref) / (nperm + 1)

    # Create result dictionary
    result = {
        'method': test.__name__,
        'statistic': float(K_ref),
        'permutations': nperm,
        'pvalue': pvalue
    }

    return result

anova_interp = {
    False: 'Significant',
    True: 'Not significant'
}

def anova1(data, rvar, cvar, order=order, var_fallback='perm', alpha=alpha, log_trans=False, show_QQplot=show_QQ_plot,
           plot_residuals=False):
    l = [data.loc[(data[cvar] == sample), rvar] for sample in order]

    if log_trans:
        data[rvar] = np.log(data[rvar])

    res = stat()
    res.anova_stat(df=data, res_var=rvar,
                   anova_model=f'{rvar} ~ C({cvar})')

    if show_QQplot:
        sm.qqplot(res.anova_std_residuals, line='45')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Standardized Residuals")
        plt.show()

    if plot_residuals:
        sns.histplot(res.anova_std_residuals)
        plt.show()

    SW, SW_pvalue = stats.shapiro(res.anova_std_residuals)
    is_norm = (SW_pvalue >= alpha)
    print(f'NORMALITY OF RESIDUALS (Shapiro) : {anova_interp[is_norm]}, p={SW_pvalue}')

    if is_norm :
        BART, BART_pvalue = stats.bartlett(*l)
        is_eq_var = (BART_pvalue >= alpha)

        print(f'EQUALITY OF VARIANCES (Bartlett) : {anova_interp[is_eq_var]}, p={BART_pvalue}')
        if is_eq_var :
            anova, anova_p = f_oneway(l, alpha=alpha)
            print(f'ONE WAY ANOVA (F) : {anova_interp[anova]}, p={anova_p}')
            if not anova :
                print(f'TUKEY\'S HSD PAIRWISE GROUP COMP :')
                result = tukey(l)
        else :
            anova, anova_p = kruskal(l, alpha=alpha)
            print(f'NON-PARAM ANOVA (Kruskal) : {anova_interp[anova]}, p={anova_p}')
            if not anova :
                result = wilcoxon(l)
    else :
        if var_fallback == 'perm' :
            resb = perm(stats.bartlett, l)
            is_eq_var, eq_var_p = (resb['pvalue'] >= 0.05), resb['pvalue']
            print(f'EQUALITY VAR (Perm Bartlett) : {is_eq_var}, p={eq_var_p}')
        elif var_fallback == 'levene' :
            is_eq_var, eq_var_p = levene(l, alpha=alpha, center='mean')
            print(f'EQUALITY VAR (Levene) : {is_eq_var}, p={eq_var_p}')
        elif var_fallback == 'brown' :
            is_eq_var, eq_var_p = levene(l, alpha=alpha, center='median')
            print(f'EQUALITY VAR (Brown-Forsythe) : {is_eq_var}, p={eq_var_p}')

        if is_eq_var :
            resf = perm(stats.f_oneway, l)
            anova, anova_p = (resf['pvalue'] >= 0.05), resf['pvalue']
            print(f'PERMUTATION ONE WAY ANOVA (F) : {anova_interp[anova]}, p={anova_p}')
            if not anova:
                result = wilcoxon(l)
        else:
            anova, anova_p = kruskal(l, alpha=alpha)
            print(f'NON-PARAM ANOVA (Kruskal) : {anova_interp[anova]}, p={anova_p}')
            if not anova:
                result = wilcoxon(l)

    if anova :
        result = None
    print("")
    return anova, result

def sign_stars(pvalue) :
    if pvalue < 0.0001:
        return '****'
    if 0.0001 <= pvalue < 0.001:
        return '***'
    if 0.001 <= pvalue < 0.01 :
        return '**'
    if 0.01 <= pvalue < 0.05 :
        return '*'
    return 'ns'

if show_plots:
    repeat_colors = ['#f58228', '#596eff', '#34d333']
    start = time.time()

    fig, ax = plt.subplots(figsize=(8, 3), dpi=500)
    ax.remove() # remove outer border lines
    fig.patch.set_visible(False)
    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1.6, 1])  # Adjust these ratios as needed
    ax2 = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    plt.tight_layout()

    # =========================================================================
    # =========================================================================
    # Conjugation efficiency
    # =========================================================================
    # =========================================================================

    results['conjugation_efficiency'] = results['conjugation_efficiency']*100

    same_color_palette = ['#4f4f4f', '#4f4f4f', '#4f4f4f']

    sns.boxplot(data=results, y='conjugation_efficiency', x='sample', hue='day', orient='v', color='lightblue', order=order, ax=ax2, palette=repeat_colors, legend=False, width=0.8, showfliers=False)

    sns.stripplot(data=results, y='conjugation_efficiency', x='sample', hue='day', dodge=True, orient='v', order=order, ax=ax2, palette=same_color_palette, legend=False, alpha=0.8, jitter=False)

    ax2.set_title('Conjugation efficiency (%)', fontsize=12)
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_xticklabels([labels[sample] for sample in order])

    ylim_low, ylim_max = ax2.get_ylim()
    offset = 2.5

    # =========================================================================
    # =========================================================================
    # Exclusion index
    # =========================================================================
    # =========================================================================

    summary = pd.DataFrame(columns=['day', 'sample', 'sum_donors', 'sum_recipients', 'sum_transconjugants'])

    for day in results['day'].unique():
        for sample in order :
            sum_donors = results[(results['day'] == day) & (results['sample'] == sample)]['total_donors'].sum()
            sum_recipients = results[(results['day'] == day) & (results['sample'] == sample)]['total_recipients'].sum()
            sum_transconjugants = results[(results['day'] == day) & (results['sample'] == sample)]['total_transconjugants'].sum()
            new_df = pd.DataFrame({'day': [day], 'sample': [sample], 'sum_donors': [sum_donors], 'sum_recipients': [sum_recipients], 'sum_transconjugants': [sum_transconjugants]})
            summary = pd.concat([summary, new_df], ignore_index=True)

    summary['ratio'] = summary['sum_donors']/(summary['sum_recipients'] + summary['sum_transconjugants'])
    summary['conjugation_efficiency'] = summary['sum_transconjugants']/(summary['sum_transconjugants'] + summary['sum_recipients'])

    TraTs = [e for e in order if e != 'Empty']
    exclusion_indices = pd.DataFrame(columns=['day', 'TraT', 'Empty_conj_eff', 'TraT_conj_eff'])

    for day in results['day'].unique():
        Empty_conj_eff = summary[(summary['day'] == day) & (summary['sample'] == 'Empty')]['conjugation_efficiency'].values[0]
        for TraT in TraTs :
            TraT_conj_eff = summary[(summary['day'] == day) & (summary['sample'] == TraT)]['conjugation_efficiency'].values[0]
            new_df = pd.DataFrame({'day': [day], 'TraT': [TraT], 'Empty_conj_eff': [Empty_conj_eff], 'TraT_conj_eff': [TraT_conj_eff]})
            exclusion_indices = pd.concat([exclusion_indices, new_df]) if not exclusion_indices.empty else new_df

    exclusion_indices['exclusion_index'] = exclusion_indices['Empty_conj_eff']/exclusion_indices['TraT_conj_eff']

    sns.set_style("ticks")
    sns.barplot(data=exclusion_indices, y='exclusion_index', x='TraT', order=TraTs, errorbar='sd', width=0.4, ax=ax3, facecolor='#cfcfcf', edgecolor='black', err_kws={'linewidth': 1}, capsize=0.15)

    sns.stripplot(data=exclusion_indices, y='exclusion_index', x='TraT', orient='v', hue='day', palette=repeat_colors, order=TraTs, ax=ax3, legend=False)

    ax3.set_ylabel('', fontsize=12)
    ax3.set_xlabel('')
    ax3.set_xticklabels([labels[sample] for sample in TraTs], fontsize=10)
    ax3.set_ylim(0, 20)
    ax3.set_title('Exclusion index')
    # ax3.legend(loc='lower right', bbox_to_anchor=(1.25, 0))

    if compute_stats :
        print('\n===== EXCLUSION INDICES =====')
        exclusion_pairs = [('Empty', TraT) for TraT in TraTs]
        sign, pvalues = anova1(exclusion_indices, rvar='exclusion_index', cvar='TraT', var_fallback='perm', order=TraTs)

        if not sign:
            pp = list(zip(exclusion_pairs, pvalues))
            sign_pairs = [pair for pair, pvalue in pp if pvalue < 0.05]
            sign_pvalues = [pvalue for pair, pvalue in pp if pvalue < 0.05]

            if hide_ns :
                annotator = Annotator(ax3, sign_pairs, x='TraT', y='exclusion_index', data=exclusion_indices, order=TraTs)
                if show_p_values:
                    annotator.set_custom_annotations([f'p={pvalue:.5} ({sign_stars(pvalue)})' for pvalue in sign_pvalues])
                    annotator.annotate()
                else:
                    annotator.set_pvalues_and_annotate(sign_pvalues)
            else :
                annotator = Annotator(ax3, exclusion_pairs, x='TraT', y='exclusion_index', data=exclusion_indices, order=TraTs)
                if show_p_values:
                    annotator.set_custom_annotations([f'p={pvalue:.5} ({sign_stars(pvalue)})' for pvalue in pvalues])
                    annotator.annotate()
                else:
                    annotator.set_pvalues_and_annotate(pvalues)
    if save_plot_name :
        plt.savefig(f'./{save_plot_name}.svg', transparent=True)
    else :
        plt.show()
    plt.close()