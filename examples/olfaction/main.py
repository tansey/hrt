import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from pyhrt.utils import create_folds, pretty_str
from pyhrt.continuous import calibrate_continuous
from pyhrt.discrete import calibrate_discrete
from pyhrt.hrt import hrt

def fit_forest(X, y, n_estimators=50, max_features='auto', max_depth=None,
                    min_samples_leaf=1, random_state=0):
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               max_features=max_features,
                               max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf,
                               oob_score=False,n_jobs=8,
                               random_state=random_state)
    rf.fit(X,y)
    return rf

def fit_extratrees(X, y, n_estimators=50, max_features='auto', max_depth=None,
                    min_samples_leaf=1, random_state=0):
    rf = ExtraTreesRegressor(n_estimators=n_estimators,
                             max_features=max_features,
                             max_depth=max_depth,
                             min_samples_leaf=min_samples_leaf,
                             n_jobs=8,
                             random_state=random_state)
    rf.fit(X,y)
    return rf

def fit_cv(X, y, folds, fit_fn):
    models = []
    for fold_idx, fold in enumerate(folds):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold] = False
        print('\tFold {} ({} samples)'.format(fold_idx, X[mask].shape[0]))
        models.append(fit_fn(X[mask], y[mask]))
    return models

'''Simple model to do CV HRT testing'''
class CvModel:
    def __init__(self, models, folds, name):
        self.models = models
        self.folds = folds
        self.name = name

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for fold, model in zip(self.folds, self.models):
            y[fold] = model.predict(X[fold])
        return y
    
def load_olfaction():
    X = pd.read_csv('data/olfaction_x.csv', header=0, index_col=[0,1])
    Y = pd.read_csv('data/olfaction_y.csv', header=0, index_col=[0,1])
    morgan_names = pd.read_csv('data/CID_names_morgan.txt',
                                delimiter='\t',
                                header=None,
                                names=['CID','Name']).set_index('CID').groupby('CID').first()['Name']
    def rename_feature(feature):
        if feature.startswith("('morgan'"):
            CID = int(feature.split(',')[1][2:-2])
            feature = "('morgan', '{}')".format(morgan_names.loc[CID])
        return feature
    X.columns = map(lambda z: rename_feature(z), list(X.columns))
    descriptors = ['Bakery', 'Sour','Intensity','Sweet','Burnt','Pleasantness','Fish', 'Fruit',
                   'Garlic','Spices','Cold','Acid','Warm',
                   'Musky','Sweaty','Ammonia','Decayed','Wood','Grass',
                   'Flower','Chemical']

    target_features = { 'Intensity': [("('dragon', 'B03[C-S]')",0.03289660567678844),("('dragon', 'F03[C-S]')",0.012902376110810188),("('dragon', 'LLS_01')",0.010916370746646824),("('dragon', 'SpAbs_B(s)')",0.006706209036764483),("('dragon', 'SpMax8_Bh(s)')",0.006691835114322034),("('dragon', 'O-057')",0.006551562161857284),("('episuite', 'EXPaws Score (Log Kow)')",0.005351734961089217),("('dragon', 'SP04')",0.004670681123569409),("('morgan', 'Cyclopentene, 1-hexyl-')",0.004494880755320532),("('dragon', 'ATS2s')",0.004446783181875806)],
                        'Fruit': [("('morgan', 'ETHYL 3-HEXENOATE')",0.07658986007304416),("('morgan', '24851-98-7')",0.07538465183392296),("('morgan', '3,7-dimethylocta-2,6-dienyl propanoate')",0.03475454557598671),("('morgan', 'Triethyl orthoformate')",0.028357479214751468),("('morgan', '2,6-Octadiene, 1-ethoxy-3,7-dimethyl-, (2Z)-')",0.024255914447655257),("('morgan', 'Ethyl caproate')",0.016042672698632052),("('morgan', 'ETHYL LEVULINATE')",0.014240662859437097),("('morgan', 'Methyl jasmonate')",0.013894682893742363),("('dragon', 'Eig08_EA(bo)')",0.01352025963678552),("('dragon', 'Mor10s')",0.013363070779819458)],
                        'Pleasantness': [("('dragon', 'SssO')",0.05030118484254111),("('dragon', 'RDF015s')",0.042180809594980384),("('dragon', 'HGM')",0.022161398750160412),("('morgan', '3-Ethoxy-4-hydroxybenzaldehyde')",0.016147937552643148),("('dragon', 'P_VSA_MR_8')",0.013655548619716176),("('morgan', 'Decahydro-2-naphthyl formate')",0.011859138752900943),("('dragon', 'MATS7s')",0.01120595045682649),("('dragon', 'nHM')",0.010029872786012115),("('dragon', 'ATS1e')",0.009652308637353956),("('dragon', 'GATS2s')",0.009617893961215589)],
                        'Bakery': [("('morgan', '3-Hydroxy-4-methoxybenzaldehyde')",0.26291031720012004),("('morgan', 'Vanillin isobutyrate')",0.052837965366773376),("('morgan', '3-Ethoxy-4-hydroxybenzaldehyde')",0.048055649680837385),("('morgan', '2-ethoxy-4-formylphenyl acetate')",0.025809386276118014),("('morgan', '3,4-Dihydroxybenzaldehyde')",0.023640538736221364),("('morgan', '4-Formyl-2-methoxyphenyl acetate')",0.02319129168623981),("('morgan', 'Imidazole-2-carboxaldehyde')",0.018580189217960185),("('dragon', 'R7e+')",0.017912056868859483),("('morgan', 'ETHYL ISOVALERATE')",0.012290742913671228),("('dragon', 'SM05_AEA(ri)')",0.009926550098053275)],
                        'Sweet': [("('morgan', '3-Ethoxy-4-hydroxybenzaldehyde')",0.06731075501683734),("('morgan', 'Cyclopentenyl propionate musk')",0.029622029544761765),("('morgan', 'DIETHYL MALATE')",0.02756585908859553),("('morgan', 'ETHYL 3-HEXENOATE')",0.022407736546081147),("('morgan', 'Ethyl pentanoate')",0.018319456103565675),("('dragon', 'CATS2D_04_AL')",0.017912043806820356),("('morgan', '3-Hydroxy-4-methoxybenzaldehyde')",0.01787603852486889),("('dragon', 'SssO')",0.01632542534582108),("('morgan', '2-ethoxy-4-formylphenyl acetate')",0.012793225019581026),("('morgan', '24851-98-7')",0.010099396073527323)],
                        'Fish': [("('dragon', 'P_VSA_m_4')",0.10096158475346095),("('dragon', 'X4Av')",0.03756445029342252),("('morgan', 'tryptamine')",0.03577975631573124),("('dragon', 'R3p+')",0.02359125526253254),("('dragon', 'SssS')",0.023160506264511483),("('morgan', 'bis(1-mercaptopropyl) sulfide')",0.01965424866000371),("('morgan', '2-Phenylethyl isothiocyanate')",0.015396572727283736),("('dragon', 'G(O..S)')",0.014394583423318572),("('morgan', '3-PENTANOL')",0.013112640473365874),("('morgan', 'PIPERIDINE')",0.01303157468285042)],
                        'Garlic': [("('dragon', 'HATS3p')",0.1846354525225861),("('dragon', 'R3p+')",0.10096514603329446),("('dragon', 'P_VSA_m_4')",0.03982036768295713),("('dragon', 'Mor05m')",0.03832097795115171),("('dragon', 'S-107')",0.024582235765850176),("('dragon', 'X3Av')",0.019681535302331472),("('dragon', 'R1p+')",0.017698280156502162),("('dragon', 'Eig05_AEA(ri)')",0.013915907011830727),("('dragon', 'Psi_e_0d')",0.012156640057946432),("('dragon', 'VE1_Dz(v)')",0.011675465019818568)],
                        'Spices': [("('morgan', 'Verdoracine')",0.043135578208800006),("('morgan', 'GAMMA-TERPINENE')",0.04084358411116229),("('morgan', 'safrole')",0.018270245758610734),("('morgan', 'Xanthorrhizol')",0.017797525614917785),("('morgan', 'Nootkatin')",0.011028803891254083),("('dragon', 'Eig11_AEA(ri)')",0.009874809745830961),("('dragon', 'HATS3p')",0.008680594305347565),("('morgan', 'M-CYMENE')",0.00848340759583751),("('morgan', 'Bis(methylthio)methane')",0.008126128647714564),("('morgan', 'Thymol acetate')",0.007542972380808742)],
                        'Cold': [("('morgan', 'BETA-TERPINEOL')",0.016794970816477066),("('dragon', 'Mor14s')",0.01328844166943731),("('morgan', 'Ledol')",0.012534402293043524),("('morgan', '1-Phenylethyl propionate')",0.01146568947386984),("('dragon', 'R5i')",0.010037199954353334),("('dragon', 'Eig11_EA(ed)')",0.008755188411336394),("('morgan', '9-Decenyl acetate')",0.008322414723432642),("('morgan', 'Verbanol')",0.008012751983270325),("('morgan', 'Globulol')",0.006830501715758576),("('morgan', '(2R)-2-(3-Methylbut-2-enyl)-2,3-dihydronaphthalene-1,4-dione')",0.006525317315268867)],
                        'Sour': [("('dragon', 'SpMAD_EA(dm)')",0.04217375697090651),("('morgan', 'butyric acid')",0.034253276549856285),("('morgan', 'sulfur dioxide')",0.02593647952605923),("('dragon', 'Mor13m')",0.0250764615629962),("('dragon', 'GATS2e')",0.023356418016479825),("('morgan', 'citric acid')",0.013231454094076542),("('morgan', 'Citral')",0.010296665183875764),("('dragon', 'H0m')",0.009426570966565455),("('dragon', 'G3m')",0.009293236304203267),("('morgan', '4-PENTENOIC ACID')",0.008430981289563485)],
                        'Burnt': [("('morgan', 'Difurfuryl sulfide')",0.09993743587083961),("('dragon', 'F04[C-S]')",0.07959893660801082),("('dragon', 'HATS3v')",0.03244534821158102),("('dragon', 'B03[O-S]')",0.02912747383169666),("('morgan', '2-Methyl-1,3-dithiolane')",0.017221385326391385),("('morgan', '2,3-Lutidine')",0.015744932622857234),("('dragon', 'R4p+')",0.013700505311777012),("('morgan', 'Ethyl 3-(furfurylthio)propionate')",0.013425824222195648),("('dragon', 'Mor08s')",0.012922477761048573),("('morgan', 'Pyrazineethanethiol')",0.012576996425096451)],
                        'Acid': [("('dragon', 'ATSC2s')",0.009054151469527379),("('dragon', 'Mor07m')",0.008990598461625707),("('morgan', '2-(1-mercaptoethyl)furan')",0.008802669493727374),("('dragon', 'P1p')",0.008619631145436773),("('morgan', '2-Pentanoylfuran')",0.007623856723363095),("('dragon', 'CATS2D_04_AL')",0.0064659193037449065),("('dragon', 'AVS_B(p)')",0.006232841202969455),("('dragon', 'SM4_B(s)')",0.006119375100393725),("('morgan', '2-ethyl-3-methylpyrrole')",0.005788503316338784),("('dragon', 'Mor15p')",0.005558630310052571)],
                        'Warm': [("('dragon', 'Mor17s')",0.06467949569422662),("('morgan', '3-Ethoxy-4-hydroxybenzaldehyde')",0.043069533929791015),("('morgan', 'curcumin')",0.011293074553762286),("('morgan', 'ETHYL ISOVALERATE')",0.008950060303772251),("('dragon', 'R6e+')",0.008212259860832911),("('morgan', 'Ethyl 3-hydroxybutyrate')",0.006927422093103866),("('dragon', 'SpMax1_Bh(m)')",0.006086314491298995),("('dragon', 'Mor15p')",0.004903203904336964),("('dragon', 'R1m')",0.004350796266079133),("('morgan', 'PIPERONAL')",0.004216824403583891)],
                        'Musky': [("('dragon', 'GATS2e')",0.051940143371192904),("('dragon', 'Mor08m')",0.02508986567470639),("('dragon', 'GATS5s')",0.013766648721819852),("('morgan', 'beta-alanine')",0.011718878479721787),("('dragon', 'SpMax5_Bh(s)')",0.009268179848893185),("('dragon', 'GATS2s')",0.008565326386045461),("('dragon', 'SssO')",0.00840482330288631),("('dragon', 'Mor15p')",0.006940063245195483),("('dragon', 'Mor21m')",0.0069378010512288635),("('dragon', 'Mor08s')",0.006649003628800438)],
                        'Sweaty': [("('dragon', 'GATS2e')",0.07602611720950699),("('morgan', 'butyric acid')",0.020920028831498207),("('morgan', 'Ipsenol')",0.019905798229069564),("('dragon', 'CATS2D_01_AN')",0.015262044764950699),("('morgan', '2-Methyl-4-pentenoic acid')",0.013858280089309075),("('morgan', 'HOTRIENOL')",0.013831155693287928),("('dragon', 'GATS2s')",0.01344969737739305),("('dragon', 'GATS5s')",0.012662097260087008),("('morgan', 'ISOVALERIC ACID')",0.01208235075313917),("('dragon', 'GATS5m')",0.010457102137766969)],
                        'Ammonia': [("('dragon', 'SssO')",0.022603139975927982),("('dragon', 'R3u')",0.01168296150199855),("('morgan', '31704-80-0')",0.011557062914966),("('morgan', '4-Methylnonanoic acid')",0.01145952656754807),("('dragon', 'F02[C-O]')",0.010115111006216312),("('morgan', 'p-Tolyl phenylacetate')",0.009591529750164501),("('dragon', 'IC3')",0.007645697905869636),("('morgan', '(+)-Cuparene')",0.007570519696658494),("('morgan', nan)",0.007467339832158917),("('morgan', 'HEXANOIC ACID')",0.007077444199938891)],
                        'Decayed': [("('dragon', 'P_VSA_m_4')",0.07261117298444275),("('dragon', 'Mor07p')",0.0368579963259016),("('morgan', 'Bis(methylthio)methane')",0.03213578615642428),("('morgan', 'BDBM136314')",0.02923821594486541),("('dragon', 'SM09_EA(dm)')",0.023800407966938133),("('dragon', 'SM05_EA(dm)')",0.022220041553960557),("('morgan', '1-HEXEN-3-ONE')",0.016971890415801764),("('dragon', 'SM07_EA(dm)')",0.01620129142421609),("('morgan', '4-PENTENOIC ACID')",0.013888232883950524),("('dragon', 'Mor13m')",0.013176297304882862)],
                        'Wood': [("('morgan', '2,6-Dimethyl-4-ethylpyridine')",0.0245383377539105),("('morgan', '2-Ethyl-3,5-dimethylpyridine')",0.023903774039119693),("('morgan', '2,4,6-Trimethylpyridine')",0.010706255245077095),("('morgan', '2,3,5-Trimethylpyrazine')",0.009297119975614795),("('dragon', 'Mor28s')",0.00919417242972841),("('morgan', '10-UNDECENOIC ACID')",0.009041271662798108),("('morgan', '2,5-Dimethyl-3-isobutylpyrazine')",0.00860821066887901),("('episuite', 'Estimated MP (oC)')",0.007635470526359774),("('morgan', 'linoleic acid')",0.006463901976142919),("('morgan', 'Triethylpyrazine')",0.00610738316680191)],
                        'Grass': [("('morgan', 'cis-3-Hexenyl isovalerate')",0.08992975382568573),("('morgan', 'cis-3-Hexenyl isobutyrate')",0.04430163787869951),("('morgan', '1,1-Dimethoxynon-2-yne')",0.02306262120776053),("('morgan', 'cis-3-Hexenyl butyrate')",0.01891784806481749),("('morgan', '24168-70-5')",0.017315982542245832),("('morgan', '3-Hexenyl 2-methylbutyrate')",0.016831291153703666),("('morgan', 'cis-3-Hexenyl angelate')",0.010244825395793676),("('morgan', '1-(2,2-Dimethoxyethoxy)hexane')",0.0093118606685875),("('morgan', 'Methyl jasmonate')",0.007920269162339314),("('dragon', 'MEcc')",0.007555915273349699)],
                        'Flower': [("('morgan', 'Phenethyl pivalate')",0.02915308741651586),("('dragon', 'H_D/Dt')",0.02269784556045422),("('dragon', 'SpMax4_Bh(m)')",0.015556722814415358),("('dragon', 'JGI6')",0.014921786500494836),("('dragon', 'GATS4e')",0.013993747263218),("('morgan', 'SCHEMBL77189')",0.01165997062684801),("('dragon', 'Mor21u')",0.01124468863441247),("('morgan', '2-ETHOXYNAPHTHALENE')",0.010820306805351936),("('dragon', 'piPC07')",0.009267190218170759),("('dragon', 'R7p')",0.009212893597361074)],
                        'Chemical': [("('dragon', 'TPSA(Tot)')",0.0495335755004722),("('dragon', 'ATSC2s')",0.04208498710975382),("('dragon', 'RDF020e')",0.015950659232969774),("('dragon', 'P1p')",0.014465194218116244),("('dragon', 'RDF020i')",0.013882574445195505),("('dragon', 'SM1_Dz(m)')",0.012458083114034572),("('dragon', 'SM1_Dz(Z)')",0.01224426414068774),("('dragon', 'GATS4s')",0.010954499976717342),("('morgan', '1,2,3,4-Tetrahydronaphthalene')",0.008957806765570989),("('morgan', 'Decatone')",0.007614463809887671)]
                      }
    return X, Y, descriptors, target_features


def plot_predictions(model, X, y, desc):
    from sklearn.metrics import r2_score
    plt.close()
    y_hat = model.predict(X)
    plt.scatter(y_hat, y, color='blue')
    plt.plot([min(y.min(), y_hat.min()),max(y.max(), y_hat.max())], [min(y.min(), y_hat.min()),max(y.max(), y_hat.max())], color='red', lw=3)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('{} ($r^2$={:.4f})'.format(desc, r2_score(y, y_hat)))
    plt.tight_layout()
    plt.savefig('plots/olfaction-predictions-{}.pdf'.format(desc.lower()), bbox_inches='tight')

def run_hrt(target_feature, X, y, features, model,
            pca_components=100, discrete_threshold=10,
            nbootstraps=100, nperms=5000, verbose=False):
    feature_idx = features.get_loc(target_feature)
    fmask = np.ones(X.shape[1], dtype=bool)
    fmask[feature_idx] = False
    X_transform = X[:,fmask]
    if pca_components is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        X_transform = pca.fit_transform(X_transform)
        X_transform = np.concatenate([X[:,feature_idx:feature_idx+1], X_transform], axis=1)
    nunique = np.unique(X[:,feature_idx]).shape[0]
    if nunique <= discrete_threshold:
        if verbose:
            print('Using discrete conditional')
        results = calibrate_discrete(X_transform, 0, nbootstraps=nbootstraps)
    else:
        if verbose:
            print('Using continuous conditional')
        results = calibrate_continuous(X_transform, 0, nbootstraps=nbootstraps)
    conditional = results['sampler']
    tstat = lambda X_test: ((y - model.predict(X_test))**2).mean()
    p_value = hrt(feature_idx, tstat, X, nperms=nperms,
                        conditional=conditional,
                        lower=conditional.quantiles[0],
                        upper=conditional.quantiles[1])['p_value']
    return p_value

def load_or_fit_model(descriptor, X, Y):
    nfolds = 10
    y = Y[descriptor]
    y = y[y.notnull()]
    x = X.loc[y.index].values
    y = y.values
    
    model_path = 'data/{}.pt'.format(descriptor)
    if os.path.exists(model_path):
        forest_model = joblib.load(model_path)
    else:
        print('Fitting {}'.format(descriptor))
        folds = create_folds(x, nfolds)
        if descriptor=='Intensity':
            forest_model = CvModel(fit_cv(x, y, folds, fit_extratrees), folds, 'ExtraTrees')
        else:
            forest_model = CvModel(fit_cv(x, y, folds, fit_forest), folds, 'RandomForest')

        plot_predictions(forest_model, x, y, descriptor)
        joblib.dump(forest_model, model_path)
    
    return x, y, forest_model

def get_model_weights(cv_model):
    return np.mean([m.feature_importances_ for m in cv_model.models], axis=0)

if __name__ == '__main__':
    print('Loading olfaction data')
    np.random.seed(42)
    X, Y, descriptors, target_features = load_olfaction()
    features = X.columns
    results = {}
    for desc in descriptors:
        # Get the model and data specifically for this descriptor class
        x, y, forest_model = load_or_fit_model(desc, X, Y)
        
        results[desc] = {}
        for target_feature, importance in target_features[desc]:
            p_value = run_hrt(target_feature, x, y, features, forest_model)
            results[desc][target_feature] = (importance, p_value)
        for rank, (target_feature, importance) in enumerate(target_features[desc]):
            importance, p_value = results[desc][target_feature]
            print('{}. {} importance={:.4f} p={:.4f}'.format(rank+1, target_feature.replace('\'',''), importance, p_value))

    for desc in descriptors:
        print(desc)
        for rank, (target_feature, importance) in enumerate(target_features[desc]):
            importance, p_value = results[desc][target_feature]
            print('{}. {} importance={:.4f} p={:.4f}'.format(rank+1, target_feature.replace('\'',''), importance, p_value))
        print('')


