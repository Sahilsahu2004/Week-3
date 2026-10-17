import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
from matplotlib.ticker import FuncFormatter
np.random.seed(42)
URL="https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
raw=pd.read_csv(URL); df=raw.drop_duplicates().copy()
df["obese"]=np.where(df.bmi>=30,"Obese (BMI ≥ 30)","Not obese (BMI < 30)")
df["log_charges"]=np.log(df.charges)
sns.set_theme(style="whitegrid"); PCOL={"yes":"#D1495B","no":"#2E86AB"}
money=FuncFormatter(lambda v,_:f"${v:,.0f}")
R={"n_raw":len(raw),"n":len(df),"dups":int(raw.duplicated().sum()),"missing":int(raw.isnull().sum().sum())}
R["desc"]=df[["age","bmi","children","charges"]].describe().round(2).to_dict()
R["counts"]={"smoker":df.smoker.value_counts().to_dict(),"sex":df.sex.value_counts().to_dict(),"region":df.region.value_counts().to_dict()}
R["charges_skew"]=round(float(df.charges.skew()),2); R["logcharges_skew"]=round(float(df.log_charges.skew()),2)

def cohen_d(a,b):
    na,nb=len(a),len(b); sp=np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2)); return (a.mean()-b.mean())/sp
def boot_ci(a,b,n=5000):
    d=[np.random.choice(a,len(a)).mean()-np.random.choice(b,len(b)).mean() for _ in range(n)]; return np.percentile(d,[2.5,97.5])
def welch_ci(a,b):
    va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b); se=np.sqrt(va+vb); dfw=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)); d=a.mean()-b.mean(); t=stats.t.ppf(.975,dfw); return d-t*se,d+t*se

# ---------- assumptions
sm_,ns_=df[df.smoker=="yes"].charges,df[df.smoker=="no"].charges
R["shapiro_raw_smoker"]=[round(float(x),4) for x in stats.shapiro(sm_)]; R["shapiro_raw_non"]=[round(float(x),4) for x in stats.shapiro(ns_)]
R["shapiro_log_smoker"]=[round(float(x),4) for x in stats.shapiro(np.log(sm_))]; R["shapiro_log_non"]=[round(float(x),4) for x in stats.shapiro(np.log(ns_))]
R["levene_smoker"]=[round(float(x),6) for x in stats.levene(sm_,ns_)]
R["levene_smoker_log"]=[round(float(x),4) for x in stats.levene(np.log(sm_),np.log(ns_))]

# ---------- H1 smoker vs non (Welch)
t,pv=stats.ttest_ind(sm_,ns_,equal_var=False)
p_one=stats.ttest_ind(sm_,ns_,equal_var=False,alternative="greater").pvalue
lo,hi=welch_ci(sm_,ns_); bl,bh=boot_ci(sm_.values,ns_.values)
u,pu=stats.mannwhitneyu(sm_,ns_,alternative="two-sided")
R["H1"]=dict(mean_s=round(sm_.mean(),2),mean_n=round(ns_.mean(),2),med_s=round(sm_.median(),2),med_n=round(ns_.median(),2),sd_s=round(sm_.std(),2),sd_n=round(ns_.std(),2),n_s=len(sm_),n_n=len(ns_),
 t=round(float(t),2),p=float(pv),p_one=float(p_one),ci=[round(lo,0),round(hi,0)],boot=[round(bl,0),round(bh,0)],d=round(cohen_d(sm_,ns_),2),U=float(u),pU=float(pu),
 ratio=round(sm_.mean()/ns_.mean(),2),power=float(TTestIndPower().power(effect_size=abs(cohen_d(sm_,ns_)),nobs1=len(sm_),ratio=len(ns_)/len(sm_),alpha=.05)))
tl,pl=stats.ttest_ind(np.log(sm_),np.log(ns_),equal_var=False); R["H1"]["t_log"]=round(float(tl),2); R["H1"]["p_log"]=float(pl)
R["H1"]["geo_ratio"]=round(float(np.exp(np.log(sm_).mean()-np.log(ns_).mean())),2)

# ---------- H2 obese vs not
ob,nb=df[df.bmi>=30].charges,df[df.bmi<30].charges
t2,p2=stats.ttest_ind(ob,nb,equal_var=False); lo2,hi2=welch_ci(ob,nb); u2,pu2=stats.mannwhitneyu(ob,nb)
R["H2"]=dict(mean_o=round(ob.mean(),2),mean_n=round(nb.mean(),2),n_o=len(ob),n_n=len(nb),t=round(float(t2),2),p=float(p2),ci=[round(lo2,0),round(hi2,0)],d=round(cohen_d(ob,nb),2),pU=float(pu2))
strat={}
for s in ["no","yes"]:
    a=df[(df.smoker==s)&(df.bmi>=30)].charges; b=df[(df.smoker==s)&(df.bmi<30)].charges
    tt,pp=stats.ttest_ind(a,b,equal_var=False); l,h=welch_ci(a,b)
    strat[s]=dict(mean_o=round(a.mean(),0),mean_n=round(b.mean(),0),n_o=len(a),n_n=len(b),t=round(float(tt),2),p=float(pp),ci=[round(l,0),round(h,0)],d=round(cohen_d(a,b),2))
R["H2_strat"]=strat
# share of smokers among obese vs not
R["H2_smoker_share"]={"obese":round(float((df[df.bmi>=30].smoker=="yes").mean()*100),1),"not":round(float((df[df.bmi<30].smoker=="yes").mean()*100),1)}

# ---------- H3 region ANOVA
groups=[g.charges.values for _,g in df.groupby("region")]
F,pF=stats.f_oneway(*groups); H,pH=stats.kruskal(*groups); lev=stats.levene(*groups)
glog=[g.log_charges.values for _,g in df.groupby("region")]; Fl,pFl=stats.f_oneway(*glog)
ssb=sum(len(g)*(g.mean()-df.charges.mean())**2 for g in groups); sst=((df.charges-df.charges.mean())**2).sum()
tuk=pairwise_tukeyhsd(df.charges,df.region)
tdf=pd.DataFrame(tuk._results_table.data[1:],columns=tuk._results_table.data[0])
R["H3"]=dict(F=round(float(F),2),p=float(pF),H=round(float(H),2),pH=float(pH),eta2=round(float(ssb/sst),4),levene=[round(float(lev[0]),3),round(float(lev[1]),3)],F_log=round(float(Fl),2),p_log=float(pFl),
 means=df.groupby("region").charges.mean().round(0).to_dict(),tukey=tdf.astype(str).values.tolist())
# smokers share by region (confounding)
R["H3"]["smoker_share"]=(df.groupby("region").smoker.apply(lambda s:(s=="yes").mean()*100).round(1)).to_dict()
# region ANOVA among non-smokers
ns=df[df.smoker=="no"]; Fn,pn=stats.f_oneway(*[g.charges.values for _,g in ns.groupby("region")]); R["H3"]["F_nonsmoker"]=round(float(Fn),2); R["H3"]["p_nonsmoker"]=float(pn)

# ---------- H4 chi2 smoker x sex
ct=pd.crosstab(df.sex,df.smoker); chi,pc,dof,exp=stats.chi2_contingency(ct,correction=False)
n=ct.values.sum(); V=np.sqrt(chi/(n*(min(ct.shape)-1)))
OR,pf=stats.fisher_exact(ct.values)
R["H4"]=dict(ct=ct.to_dict(),chi=round(float(chi),2),p=float(pc),dof=int(dof),V=round(float(V),3),exp=pd.DataFrame(exp,index=ct.index,columns=ct.columns).round(1).to_dict(),
 pct={s:round(float((df[df.sex==s].smoker=="yes").mean()*100),1) for s in ["male","female"]},OR=round(float(OR),2))
# OR CI via log method
a,b_,c_,d_=ct.loc["female","no"],ct.loc["female","yes"],ct.loc["male","no"],ct.loc["male","yes"]
se=np.sqrt(1/a+1/b_+1/c_+1/d_); orr=(d_*a)/(b_*c_); R["H4"]["OR_male_vs_female"]=round(float(orr),2); R["H4"]["OR_ci"]=[round(float(np.exp(np.log(orr)-1.96*se)),2),round(float(np.exp(np.log(orr)+1.96*se)),2)]

# ---------- H5 age correlation
r,pr=stats.pearsonr(df.age,df.charges); rho,prho=stats.spearmanr(df.age,df.charges)
rn,pn_=stats.pearsonr(ns.age,ns.charges)
def rci(r,n): z=np.arctanh(r); s=1/np.sqrt(n-3); return [round(float(np.tanh(z-1.96*s)),3),round(float(np.tanh(z+1.96*s)),3)]
R["H5"]=dict(r=round(float(r),3),p=float(pr),ci=rci(r,len(df)),rho=round(float(rho),3),prho=float(prho),r_non=round(float(rn),3),p_non=float(pn_),ci_non=rci(rn,len(ns)),
 r_s=round(float(stats.pearsonr(df[df.smoker=="yes"].age,df[df.smoker=="yes"].charges)[0]),3))
lr=stats.linregress(ns.age,ns.charges); R["H5"]["slope_non"]=round(float(lr.slope),1); R["H5"]["intercept_non"]=round(float(lr.intercept),0)
lrs=stats.linregress(df[df.smoker=="yes"].age,df[df.smoker=="yes"].charges); R["H5"]["slope_s"]=round(float(lrs.slope),1)

# ---------- H6 regression
m=smf.ols("log_charges ~ age + bmi + children + C(smoker) + C(sex) + C(region)",data=df).fit()
R["OLS"]=dict(r2=round(float(m.rsquared),3),adj=round(float(m.rsquared_adj),3),F=round(float(m.fvalue),1),pF=float(m.f_pvalue),
 coef={k:dict(b=round(float(v),4),p=float(m.pvalues[k]),lo=round(float(m.conf_int().loc[k,0]),4),hi=round(float(m.conf_int().loc[k,1]),4),pct=round(float((np.exp(v)-1)*100),1)) for k,v in m.params.items()})
m2=smf.ols("charges ~ age + bmi + children + C(smoker) + C(sex) + C(region)",data=df).fit(); R["OLS_raw_smoker"]=[round(float(m2.params["C(smoker)[T.yes]"]),0),round(float(m2.conf_int().loc["C(smoker)[T.yes]",0]),0),round(float(m2.conf_int().loc["C(smoker)[T.yes]",1]),0)]
sw=stats.shapiro(m.resid); R["OLS"]["resid_shapiro"]=[round(float(sw[0]),3),float(sw[1])]
df.to_csv("insurance_clean.csv",index=False)
# multiple testing
ps=[R["H1"]["p"],R["H2"]["p"],R["H3"]["p"],R["H4"]["p"],R["H5"]["p"]]; R["bonf_alpha"]=0.01

# ================= FIGURES =================
# F1 distributions
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
for s,lab in [("no","Non-smokers"),("yes","Smokers")]:
    d=df[df.smoker==s].charges; sns.histplot(d,bins=35,kde=True,color=PCOL[s],alpha=.45,label=f"{lab} (mean ${d.mean():,.0f})",ax=ax[0]); ax[0].axvline(d.mean(),color=PCOL[s],ls="--",lw=2)
ax[0].xaxis.set_major_formatter(money); ax[0].set_title("Charges by smoking status",fontweight="bold"); ax[0].set_xlabel("Annual medical charges"); ax[0].legend()
sns.boxplot(data=df,x="smoker",y="charges",palette=PCOL,ax=ax[1],order=["no","yes"]); sns.stripplot(data=df.sample(400,random_state=1),x="smoker",y="charges",color="k",alpha=.25,size=2.5,ax=ax[1],order=["no","yes"])
ax[1].yaxis.set_major_formatter(money); ax[1].set_xticklabels(["Non-smoker","Smoker"]); ax[1].set_xlabel(""); ax[1].set_title("Spread and outliers",fontweight="bold"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("s1_dist.png",dpi=160); plt.close()
# F2 QQ
fig,ax=plt.subplots(1,3,figsize=(14,4.4))
stats.probplot(df.charges,plot=ax[0]); ax[0].set_title("Q-Q plot: raw charges\n(curved = not normal)",fontweight="bold")
stats.probplot(df.log_charges,plot=ax[1]); ax[1].set_title("Q-Q plot: log(charges)\n(closer to a straight line)",fontweight="bold")
stats.probplot(m.resid,plot=ax[2]); ax[2].set_title("Q-Q plot: regression residuals",fontweight="bold")
for a_ in ax: a_.get_lines()[0].set_markersize(3); a_.get_lines()[1].set_color("#D1495B")
plt.tight_layout(); plt.savefig("s2_qq.png",dpi=160); plt.close()
# F3 mean CI plot H1+H2
groups_ci=[]
for lab,mask in [("Non-smoker",df.smoker=="no"),("Smoker",df.smoker=="yes"),("Not obese",df.bmi<30),("Obese",df.bmi>=30),
  ("Non-smoker\nnot obese",(df.smoker=="no")&(df.bmi<30)),("Non-smoker\nobese",(df.smoker=="no")&(df.bmi>=30)),("Smoker\nnot obese",(df.smoker=="yes")&(df.bmi<30)),("Smoker\nobese",(df.smoker=="yes")&(df.bmi>=30))]:
    x=df[mask].charges; se=x.std()/np.sqrt(len(x)); groups_ci.append((lab,x.mean(),1.96*se,len(x)))
R["ci_table"]=[[g[0].replace("\n"," "),round(g[1],0),round(g[1]-g[2],0),round(g[1]+g[2],0),g[3]] for g in groups_ci]
fig,ax=plt.subplots(figsize=(11,5))
cols=["#2E86AB","#D1495B","#7FB069","#E6AA68","#2E86AB","#7FB069","#D1495B","#E6AA68"]
cols=["#2E86AB","#D1495B","#8AA6C1","#C9A66B","#2E86AB","#2E86AB","#D1495B","#D1495B"]
for i,(lab,mu,h,nn) in enumerate(groups_ci):
    ax.errorbar(i,mu,yerr=h,fmt="o",color=cols[i],capsize=6,ms=9,lw=2.5); ax.text(i,mu+h+900,f"${mu:,.0f}",ha="center",fontsize=9,fontweight="bold")
ax.set_xticks(range(len(groups_ci))); ax.set_xticklabels([g[0] for g in groups_ci],fontsize=9); ax.yaxis.set_major_formatter(money)
ax.axvline(3.5,color="#aaa",ls="--"); ax.set_ylim(0,48000); ax.set_ylabel("Mean annual charges (95% CI)")
ax.set_title("Mean charges with 95% confidence intervals: smoking dominates, obesity matters mainly for smokers",fontweight="bold",fontsize=11.5,loc="left")
plt.tight_layout(); plt.savefig("s3_ci.png",dpi=160); plt.close()
# F4 region
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
order=df.groupby("region").charges.mean().sort_values().index
sns.boxplot(data=df,x="region",y="charges",order=order,palette="Set2",ax=ax[0]); ax[0].yaxis.set_major_formatter(money); ax[0].set_title(f"Charges by region (ANOVA p = {pF:.3f})",fontweight="bold"); ax[0].set_xlabel(""); ax[0].set_ylabel("")
sh=df.groupby("region").smoker.apply(lambda s:(s=="yes").mean()*100).reindex(order)
sns.barplot(x=sh.index,y=sh.values,palette="Set2",ax=ax[1]); ax[1].set_title("Share of smokers by region (%)",fontweight="bold"); ax[1].set_xlabel(""); ax[1].set_ylabel("% smokers")
for i,v in enumerate(sh.values): ax[1].text(i,v+.3,f"{v:.1f}%",ha="center",fontweight="bold")
plt.tight_layout(); plt.savefig("s4_region.png",dpi=160); plt.close()
# F5 chi-square
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
pc_=pd.crosstab(df.sex,df.smoker,normalize="index")*100
pc_[["no","yes"]].plot(kind="bar",stacked=True,color=[PCOL["no"],PCOL["yes"]],ax=ax[0],rot=0,width=.55)
for i,s in enumerate(pc_.index): ax[0].text(i,pc_.loc[s,"no"]+pc_.loc[s,"yes"]/2,f"{pc_.loc[s,'yes']:.1f}%\nsmokers",ha="center",va="center",color="white",fontweight="bold")
ax[0].set_ylabel("% of group"); ax[0].set_xlabel(""); ax[0].set_title(f"Smoking by sex (chi-square p = {pc:.4f})",fontweight="bold"); ax[0].legend(["Non-smoker","Smoker"],loc="lower right")
o=ct.values; e=exp; lab=["Female\nnon-smoker","Female\nsmoker","Male\nnon-smoker","Male\nsmoker"]
x=np.arange(4); ax[1].bar(x-.18,o.flatten(),.36,label="Observed",color="#1F3A5F"); ax[1].bar(x+.18,e.flatten(),.36,label="Expected if independent",color="#B8C4D6")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab,fontsize=9); ax[1].legend(); ax[1].set_title("Observed vs expected counts",fontweight="bold")
plt.tight_layout(); plt.savefig("s5_chi.png",dpi=160); plt.close()
# F6 scatter age
fig,ax=plt.subplots(figsize=(10,5.5))
for s,lab in [("no","Non-smoker"),("yes","Smoker")]:
    d=df[df.smoker==s]; sns.regplot(data=d,x="age",y="charges",scatter_kws=dict(alpha=.35,s=14,color=PCOL[s]),line_kws=dict(color=PCOL[s],lw=2.5),ax=ax,label=lab,ci=95)
ax.yaxis.set_major_formatter(money); ax.set_xlabel("Age"); ax.set_ylabel("Annual charges")
ax.set_title(f"Charges rise with age (overall r = {r:.2f}); smokers pay far more at every age",fontweight="bold",fontsize=12,loc="left"); ax.legend()
plt.tight_layout(); plt.savefig("s6_age.png",dpi=160); plt.close()
# F7 coefficient plot
cf=R["OLS"]["coef"]; names={"C(smoker)[T.yes]":"Smoker (vs non)","age":"Age (per year)","bmi":"BMI (per point)","children":"Children (per child)","C(sex)[T.male]":"Male (vs female)","C(region)[T.northwest]":"Region: northwest","C(region)[T.southeast]":"Region: southeast","C(region)[T.southwest]":"Region: southwest"}
rows=[(names[k],(np.exp(v["b"])-1)*100,(np.exp(v["lo"])-1)*100,(np.exp(v["hi"])-1)*100,v["p"]) for k,v in cf.items() if k!="Intercept"]; rows.sort(key=lambda z:z[1])
fig,ax=plt.subplots(figsize=(10,5))
for i,(nm,pt,lo_,hi_,pp) in enumerate(rows):
    col="#D1495B" if pp<.05 else "#999999"; ax.errorbar(pt,i,xerr=[[pt-lo_],[hi_-pt]],fmt="o",color=col,capsize=4,lw=2,ms=7); ax.text(hi_+6 if hi_<150 else lo_-6,i,f"{pt:+.1f}%",va="center",ha="left" if hi_<150 else "right",fontsize=9,color=col,fontweight="bold")
ax.axvline(0,color="k",lw=1); ax.set_yticks(range(len(rows))); ax.set_yticklabels([x[0] for x in rows]); ax.set_xlabel("% change in charges, holding other factors constant (95% CI)")
ax.set_title("Regression: which factors independently predict charges? (red = significant)",fontweight="bold",fontsize=11.5,loc="left"); ax.set_xlim(-30,470)
plt.tight_layout(); plt.savefig("s7_coef.png",dpi=160); plt.close()
json.dump(R,open("res.json","w"),indent=1,default=str)
print(json.dumps({k:R[k] for k in R if k not in ["desc"]},indent=1,default=str))
