import uproot
import hist
import mplhep as hep
import matplotlib.pyplot as plt

file = uproot.open("histograms_all.root")

data = file['4j2b_data'].to_hist()
ttbar = file['4j2b_ttbar'].to_hist()
single_atop = file['4j2b_single_atop_t_chan'].to_hist()
single_top = file['4j2b_single_top_t_chan'].to_hist()
single_tW = file['4j2b_single_top_tW'].to_hist()
single_top_s = file['4j2b_single_top_s_chan'].to_hist()
wjets = file['4j2b_wjets'].to_hist()
bklabels = ["ttbar","single_atop_t_chan","single_top_t_chan","single_top_tW","single_top_s_chan","wjets"]

fig,ax = plt.subplots(figsize = (10,10))
hep.style.use("CMS")
hep.cms.label("open data",data=True, lumi=2.26, year=2015)
hep.histplot(data,histtype="errorbar", color='k', capsize=4, label="Data", ax=ax)
hep.histplot([ttbar,single_atop,single_top,single_tW,single_top_s,wjets],
             stack=True, 
             histtype='fill', 
             label=bklabels, 
             sort='yield',
             ax=ax)
ax.legend(frameon=False)
ax.set_xlabel("$m_{bjj}$ [Gev]");
fig.savefig('finalplot_all.png')
plt.show()