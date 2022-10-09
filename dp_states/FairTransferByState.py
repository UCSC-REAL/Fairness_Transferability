import numpy as np
from sklearn.linear_model import LogisticRegression
import folktables
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


EPS = 1e-12
NUM_BIN = 100
GROUP = 'SEX'
GROUP_0_KEY = 1
GROUP_1_KEY = 2


ACSIncome = folktables.BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,    
    group=GROUP,
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


# load dataset
def load_dataset(state: str, year: int):
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    data = data_source.get_data(states=[state], download=True)
    features, label, group = ACSIncome.df_to_numpy(data)
    return features, np.expand_dims(label, axis=1), np.expand_dims(group, axis=1)


# preprocess
def preprocess(X, Y, A):
    data = np.concatenate((X, Y, A), axis=1)
    A = A.squeeze()
    data = data[(A == GROUP_0_KEY) | (A == GROUP_1_KEY)]
    return data[:, :-2].squeeze(), data[:,-2].squeeze(), data[:,-1].squeeze()


# equal odds
def tpr(X, Y, threshold=0.5):
    tp = ((X >= threshold) & (Y == 1)).sum()
    fn = ((X < threshold) & (Y == 1)).sum()
    return tp / (tp + fn)


def fpr(X, Y, threshold=0.5):
    fp = ((X >= threshold) & (Y == 0)).sum()
    tn = ((X < threshold) & (Y == 0)).sum()
    return fp / (fp + tn)


# equal odds disparity
def disparity(X, Y, A, tau_a, tau_b):
    return abs((X[A==GROUP_0_KEY] >= tau_a).mean() - (X[A==GROUP_1_KEY] >= tau_b).mean())


def run(sState: str, tState: str, year: int):
    # load data
    X_s, Y_s, A_s = preprocess(*load_dataset(sState, year))
    X_t, Y_t, A_t = preprocess(*load_dataset(tState, year))
    clf = LogisticRegression(solver='liblinear', fit_intercept=True).fit(X_s, Y_s)
    X_s, X_t = map(lambda x: clf.predict_proba(x)[:, 1], (X_s, X_t))

    # fairness violation
    def delta(tau_a, tau_b):
        source = disparity(X_s, Y_s, A_s, tau_a, tau_b)
        realized = disparity(X_t, Y_t, A_t, tau_a, tau_b)
        return realized - source

    # covariate shift bound
    def covariate_bound(tau_a, tau_b):
        def w(xs, xr):
            bin_s, _ = np.histogram(xs, bins=NUM_BIN, range=(0., 1.), density=True)
            bin_r, _ = np.histogram(xr, bins=NUM_BIN, range=(0., 1.), density=True)
            bin_s[bin_s==0] = EPS
            bin_r[bin_r==0] = EPS
            return np.cov(bin_r / bin_s, aweights=bin_s).item()
        return (X_s[A_s==GROUP_0_KEY] >= tau_a).mean() * w(X_s[A_s==GROUP_0_KEY], X_t[A_t==GROUP_0_KEY]) + \
            (X_s[A_s==GROUP_1_KEY] >= tau_b).mean() * w(X_s[A_s==GROUP_1_KEY], X_t[A_t==GROUP_1_KEY])

    # label shift bound
    def label_bound(tau_a, tau_b):
        p_1 = abs(Y_s[A_s==GROUP_0_KEY].mean() - Y_t[A_t == GROUP_0_KEY].mean())
        p_2 = abs(Y_s[A_s==GROUP_1_KEY].mean() - Y_t[A_t == GROUP_1_KEY].mean())
        return p_1 * abs(tpr(X_s[A_s==GROUP_0_KEY], Y_s[A_s==GROUP_0_KEY], threshold=tau_a) - fpr(X_s[A_s==GROUP_0_KEY], Y_s[A_s==GROUP_0_KEY], threshold=tau_a)) + \
            p_2 * abs(tpr(X_s[A_s==GROUP_1_KEY], Y_s[A_s==GROUP_1_KEY], threshold=tau_b) - fpr(X_s[A_s==GROUP_1_KEY], Y_s[A_s==GROUP_1_KEY], threshold=tau_b))
    
    bx, by = np.meshgrid(np.linspace(0, 1.0, 100), np.linspace(0, 1.0, 100))
    db = np.vectorize(covariate_bound)(bx, by)
    dc = np.vectorize(label_bound)(bx, by)
    dz = np.vectorize(delta)(bx, by)
    plot(bx, by, db, dz, f"covariate_shift_{sState}_{tState}_{year}")
    plot(bx, by, dc, dz, f"label_shift_{sState}_{tState}_{year}")
    print(f"covariate shift rate: {(dz < db).mean()*100:.2f}")
    print(f"label shift rate: {(dz < dc).mean()*100:.2f}")


# plot
def plot(bx, by, db, dz, name):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(bx, by, dz)
    ax.plot_surface(bx, by, db, rstride=1, cstride=1, cmap=cm.viridis)
    ax.set_xlabel('tau_g')
    ax.set_ylabel('tau_h')
    ax.set_zlabel('Change in Violation')
    ax.view_init(15, -60)
    plt.savefig(f'fig/{name}.pdf')