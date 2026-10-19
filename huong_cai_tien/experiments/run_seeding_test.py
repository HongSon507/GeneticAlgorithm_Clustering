"""
Thu nghiem: gieo mot phan quan the ban dau tu GMM/k-means (k=2..6), KHONG dung
nhan that, de tach loi "tim kiem" khoi loi "fitness". Fitness giu nguyen.

Chay: python huong_cai_tien/experiments/run_seeding_test.py
"""
import _paths, numpy as np, ga
from ga import NKHGA
from search import ls_fi
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ari
from nkcv2_affinity import AffinityNKCV2Model, build_affinity
from datasets import DATASETS

class SeededNKHGA(NKHGA):
    """Gieo mot phan quan the ban dau tu GMM/k-means (k=2..6), khong dung nhan that."""
    seeds_labels = []
    def fit(self, X):
        self._queue = list(self.seeds_labels)
        return super().fit(X)
    def _immigrant(self, model, rng):
        if getattr(self, "_queue", None):
            x = np.asarray(self._queue.pop(0), dtype=np.int64).copy()
            f = model.comp_fitness(x)
            f = ls_fi(model, x, f, rng)
            return x, f
        return super()._immigrant(model, rng)

for name in ["flame","jain","aggregation"]:
    X,y=DATASETS[name]()
    seeds=[]
    for k in range(2,7):
        seeds.append(GaussianMixture(k,covariance_type="full",random_state=k).fit_predict(X)+1)
        seeds.append(KMeans(k,n_init=3,random_state=k).fit_predict(X)+1)
    A=build_affinity(X,"ca_gmm") if name!="aggregation" else build_affinity(X,"snn")
    best_seed_ari=max(ari(y,s) for s in seeds)
    for label,beta in [("base",0.0),("NKCV-A",0.5)]:
        res=[]
        for s in [100,101,102]:
            for seeded in [False,True]:
                orig=ga.NKCV2Model
                ga.NKCV2Model=lambda Xx,K=3: AffinityNKCV2Model(Xx,K=K,beta=beta,A=A)
                try:
                    cls=SeededNKHGA if seeded else NKHGA
                    if seeded: cls.seeds_labels=seeds
                    est=cls(K=3,pop_size=60,max_gen=80,random_state=s).fit(X)
                finally: ga.NKCV2Model=orig
                res.append((seeded,ari(y,est.labels_),est.n_clusters_))
        for seeded in [False,True]:
            a=[r[1] for r in res if r[0]==seeded]; c=[r[2] for r in res if r[0]==seeded]
            print(f"{name:12}{label:7}{'gieo' if seeded else 'ngẫu nhiên':11} ARI={np.mean(a):.3f}±{np.std(a):.3f} cụm={np.mean(c):.1f}  per-seed={[round(v,3) for v in a]}", flush=True)
    print(f"{name:12}(ARI tốt nhất trong các lời giải gieo = {best_seed_ari:.3f})", flush=True)
