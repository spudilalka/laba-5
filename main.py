
# import array as arr
# import scipy.stats
# from matplotlib import pyplot as plt   
# from scipy.stats import f_gen
# #ploting our canvas   
# plt.plot([1,2,3],[4,5,1])   
# #display the graph   
# plt.show()   


# #find F critical value

# arry = arr.array('d',[])
# arry
# gg = 0

# for i in range(10):
  
#    for j in range(10):
#       gg+=1
#       arry.insert(0,gg)
    

# # for val in arry: 
# #     val = i

# for val in arry: 
#    print(val)


    

# print(scipy.stats.f.ppf(q=1-.05, dfn=6, dfd=8))
# scipy.stats.f.ppf(q=1-.05, dfn=6, dfd=8)

# # 3.5806

# import numpy as np
# from scipy.stats import f
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)


# dfn, dfd = 29, 18
# mean, var, skew, kurt = f.stats(dfn, dfd, moments='mvsk')


# x = np.linspace(f.ppf(0.01, dfn, dfd),
#                 f.ppf(0.99, dfn, dfd), 100)
# ax.plot(x, f.pdf(x, dfn, dfd),
#        'r-', lw=5, alpha=0.6, label='f pdf')


# rv = f(dfn, dfd)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


# vals = f.ppf([0.001, 0.5, 0.999], dfn, dfd)
# np.allclose([0.001, 0.5, 0.999], f.cdf(vals, dfn, dfd))
# True


# r = f.rvs(dfn, dfd, size=1000)

# ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
# ax.set_xlim([x[0], x[-1]])
# ax.legend(loc='best', frameon=False)
# plt.show()









import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# pip install -U scikit-learn

regression_line = lambda a, b: lambda x: a + (b * x)  # вызовы fn(a,b)(x)

x = np.array([1,2,3,4,5,6,7,8,9,10,11 ]).reshape((-1, 1))
y = np.array([124.9,127.1,134.0,139.1,147.3,155.0,159.8,165.4,172.5,177.4,182.1])
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
# coefficient of determination: 0.715875613747954
print('intercept:', model.intercept_)
# intercept: 5.633333333333329
print('slope:', model.coef_)
# slope: [0.54]
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
# y_pred = model.intercept_ + model.coef_ * x




# z = np.arange(0, 11, 0.01)
# plt.plot(z, z*model.coef_+model.intercept_)
# plt.scatter(x, y)
# plt.show()




# pd = plt.subplots(1, 1)

# ax = pd.DataFrame(np.array([x, y]).T).plot.scatter(0, 1, s=7)
# s  = pd.Series(range(100,150))
# df = pd.DataFrame( {0:s, 1:s.map(regression_line(model.coef_, model.intercept_))} )  
# df.plot(0, 1, legend=False, grid=True, ax=ax)
# plt.xlabel('Xi')
# plt.ylabel('Yi')
# plt.show()




import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt


# %matplotlib inline

# Raw Data
heights = np.array([1,2,3,4,5,6,7,8,9,10,11 ])
weights = np.array([124.9,127.1,134.0,139.1,147.3,155.0,159.8,165.4,172.5,177.4,182.1])


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci )

    return ax

def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b) 


x = heights
y = weights
p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients

# Statistics
n = weights.size                                           # number of observations
m = p.size                                                 # number of parameters
dof = n - m                                                # degrees of freedom
t = stats.t.ppf(0.9, n - m)                              # t-statistic; used for CI and PI bands

# Estimates of Error in Data/Model
resid = y - y_model                                        # residuals; diff. actual data from predicted values
chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

# Plotting --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Data
ax.plot(
    x, y, "o", color="#b9cfe7", markersize=8, 
    markeredgewidth=1, markeredgecolor="b", markerfacecolor="None"
)

# Fit
ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  

x2 = np.linspace(np.min(x), np.max(x), 100)
y2 = equation(p, x2)

# Confidence Interval (select one)
plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
#plot_ci_bootstrap(x, y, resid, ax=ax)
   
# Prediction Interval
pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
ax.plot(x2, y2 + pi, "--", color="0.5")

# plt.show()
# Figure Modifications --------------------------------------------------------
# Borders
ax.spines["top"].set_color("0.5")
ax.spines["bottom"].set_color("0.5")
ax.spines["left"].set_color("0.5")
ax.spines["right"].set_color("0.5")
ax.get_xaxis().set_tick_params(direction="out")
ax.get_yaxis().set_tick_params(direction="out")
ax.xaxis.tick_bottom()
ax.yaxis.tick_left() 

# Labels
plt.title("Fit Plot for Weight", fontsize="14", fontweight="bold")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.xlim(np.min(x) - 1, np.max(x) + 1)

# Custom legend
handles, labels = ax.get_legend_handles_labels()
display = (0, 1)
anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")    # create custom artists
legend = plt.legend(
    [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
    [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
    loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
)  
frame = legend.get_frame().set_edgecolor("0.5")


plt.tight_layout()


plt.show()