#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>

namespace py = pybind11;

// dict type
struct data_struct {
  std::vector<float> area;
  std::vector<int> iscrowd;
  std::vector<std::vector<float>> bbox;
  std::vector<int> ignore;
  std::vector<float> score;
  std::vector<std::vector<int>> segm_size;
  std::vector<std::string> segm_counts;
  std::vector<int64_t> id;
};

// internal prepare results
inline size_t key(int i,int j) {return (size_t) i << 32 | (unsigned int) j;}
std::unordered_map<size_t, data_struct> gts_map;
std::unordered_map<size_t, data_struct> dts_map;

// internal computeiou results
std::vector<int> evalImgs_list;
std::vector<std::vector<int64_t>> gtIgnore_list;
std::vector<std::vector<double>> dtIgnore_list;
std::vector<std::vector<double>> dtMatches_list;
std::vector<std::vector<double>> dtScores_list;


template <typename T>
std::vector<size_t> sort_indices(std::vector<T>& v) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Note > instead of < in comparator for descending sort
  std::sort(indices.begin(), indices.end(),
           [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return indices;
}

void
cpp_evaluate_img(int useCats, int maxDet,
                 std::vector<int> catIds,
                 std::vector<std::vector<double>> areaRngs,
                 std::vector<int> imgIds,
                 py::dict gts,
                 py::dict dts,
                 py::dict selfious_dict,
                 std::vector<double> iouThrs_ptr) {
  assert(useCats > 0);

  for(size_t c = 0; c < catIds.size(); c++) {
  for(size_t a = 0; a < areaRngs.size(); a++) {
  for(size_t i = 0; i < imgIds.size(); i++) {
    int catId = catIds[c];
    int imgId = imgIds[i];

    auto gtsm = &gts_map[key(imgId,catId)];
    auto dtsm = &dts_map[key(imgId,catId)];

    if((gtsm->id.size()==0) && (dtsm->id.size()==0)) {
      evalImgs_list.push_back(0);
      continue;
    }
    
    double aRng0 = areaRngs[a][0];
    double aRng1 = areaRngs[a][1];
    py::tuple k = py::make_tuple(imgId,catId);
    py::array selfious = selfious_dict[k];
    // sizes
    int T = iouThrs_ptr.size();
    int G = gtsm->id.size();
    int Do = dtsm->id.size();
    int D = (maxDet < Do) ? maxDet : Do;
    int I = selfious.request().shape[0];
    //int I = selfious_ptr.size() / Do;
    // arrays
    std::vector<size_t> gtind(G);
    std::vector<size_t> dtind(Do);
    std::vector<int> iscrowd(G);
    std::vector<double> gtm(T*G);
    std::vector<int64_t> gtIds(G);
    std::vector<int64_t> dtIds(D);
    gtIgnore_list.push_back(std::vector<int64_t>(G));
    dtIgnore_list.push_back(std::vector<double>(T*D));
    dtMatches_list.push_back(std::vector<double>(T*D));
    dtScores_list.push_back(std::vector<double>(D));
    // pointers
    auto gtIg = &gtIgnore_list.back()[0];
    auto dtIg = &dtIgnore_list.back()[0];
    auto dtm = &dtMatches_list.back()[0];
    auto dtScores = &dtScores_list.back()[0];
    // load computed ious
    auto selfious_buf = selfious.request();
    double *selfious_ptr = (double *) selfious_buf.ptr;
    // set ignores
    for (int g = 0; g < G; g++) {
      double area = gtsm->area[g];
      int64_t ignore = gtsm->ignore[g];
      if(ignore || (area<aRng0 || area>aRng1)) {
        gtIg[g] = 1;
      }
      else {
        gtIg[g] = 0;
      }
    }
    // sort dt highest score first, sort gt ignore last
    std::vector<float> ignores;
    for (int g = 0; g < G; g++) {
      auto ignore = gtIg[g];
      ignores.push_back(ignore);
    }
    auto g_indices = sort_indices(ignores);
    for (int g = 0; g < G; g++) {
      gtind[g] = g_indices[g];
    }
    std::vector<float> scores;
    for (int d = 0; d < Do; d++) {
      auto score = -dtsm->score[d];
      scores.push_back(score);
    }
    auto indices = sort_indices(scores);
    for (int d = 0; d < Do; d++) {
      dtind[d] = indices[d];
    }
    // iscrowd and ignores
    for (int g = 0; g < G; g++) {
      iscrowd[g] = gtsm->iscrowd[gtind[g]];
      gtIg[g] = ignores[gtind[g]];
    }
    // zero arrays
    for (int t = 0; t < T; t++) {
      for (int g = 0; g < G; g++) {
        gtm[t * G + g] = 0;
      }
      for (int d = 0; d < D; d++) {
        dtm[t * D + d] = 0;
        dtIg[t * D + d] = 0;
      }
    }
    // if not len(ious)==0:
    if(I != 0) {
      for (int t = 0; t < T; t++) {
        double thresh = iouThrs_ptr[t];
        for (int d = 0; d < D; d++) {
          double iou = thresh < (1-1e-10) ? thresh : (1-1e-10);
          int m = -1;
          for (int g = 0; g < G; g++) {
            // if this gt already matched, and not a crowd, continue
            if((gtm[t * G + g]>0) && (iscrowd[g]==0))
              continue;
            // if dt matched to reg gt, and on ignore gt, stop
            if((m>-1) && (gtIg[m]==0) && (gtIg[g]==1))
              break;
            // continue to next gt unless better match made
            double val = selfious_ptr[d + I * gtind[g]];
            if(val < iou)
              continue;
            // if match successful and best so far, store appropriately
            iou=val;
            m=g;
          }
          // if match made store id of match for both dt and gt
          if(m ==-1)
            continue;
          dtIg[t * D + d] = gtIg[m];
          dtm[t * D + d]  = gtsm->id[gtind[m]];
          gtm[t * G + m]  = dtsm->id[dtind[d]];
        }
      }
    }
    // set unmatched detections outside of area range to ignore
    for (int d = 0; d < D; d++) {
      float val = dtsm->area[dtind[d]];
      double x3 = (val<aRng0 || val>aRng1);
      for (int t = 0; t < T; t++) {
        double x1 = dtIg[t * D + d];
        double x2 = dtm[t * D + d];
        double res = x1 || ((x2==0) && x3);
        dtIg[t * D + d] = res;
      }
    }
    // store results for given image and category
    for (int g = 0; g < G; g++) {
      gtIds[g] = gtsm->id[gtind[g]];
    }
    for (int d = 0; d < D; d++) {
      dtIds[d] = dtsm->id[dtind[d]];
      dtScores[d] = dtsm->score[dtind[d]];
    }
    evalImgs_list.push_back(1);
  }}}
}

template <typename T>
std::vector<size_t> stable_sort_indices(std::vector<T>& v) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Note > instead of < in comparator for descending sort
  std::stable_sort(indices.begin(), indices.end(),
           [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return indices;
}

template <typename T>
std::vector<T> assemble_array(std::vector<int> Eind, std::vector<std::vector<T>>& list, size_t nrows, size_t maxDet, std::vector<size_t>& indices) {
  // i.e. dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
  // all instances of e[field] must have the same number of rows, simply append columns
  std::vector<T> q;

  // Need to get N_rows from an entry in order to compute output size
  // copy first maxDet entries from each entry -> array
  for (size_t e = 0; e < Eind.size(); ++e) {
    auto arr_ptr = list[Eind[e]];
    size_t cols = arr_ptr.size() / nrows;

    for (size_t j = 0; j < std::min(maxDet, cols); ++j) {
    for (size_t i = 0; i < nrows; ++i) {
        q.push_back(arr_ptr[i * cols + j]);
      }
    }
  }
  // now we've done that, copy the relevant entries based on indices
  std::vector<T> res(indices.size() * nrows);
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < indices.size(); ++j) {
      res[i * indices.size() + j] = q[indices[j] * nrows + i];
    }
  }

  return res;
}

std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>>
cpp_accumulate(int T, int R, int K, int A, int M, 
                    int I0, int A0,
                    std::vector<int> k_list,      // category ids
                    std::vector<int> a_list,      // area ranges
                    std::vector<int> m_list,      // max detections
                    std::vector<int> i_list,      // image ids
                    std::vector<double> recThrs) {
  std::vector<double> precision(T*R*K*A*M);
  std::vector<double> recall(T*K*A*M);
  std::vector<double> scores(T*R*K*A*M);

  int count = 0;
  for (size_t k = 0; k < k_list.size(); ++k) {
    auto k0 = k_list[k];
    auto Nk = k0 * A0 * I0;
    for (size_t a = 0; a < a_list.size(); ++a) {
      auto a0 = a_list[a];
      auto Na = a0 * I0;
      std::vector<int> Eind;
      for (size_t i = 0; i < i_list.size(); ++i) {
        if (evalImgs_list[Nk + Na + i]) {
          Eind.push_back(count); count++;
        }
      }
      for (size_t m = 0; m < m_list.size(); ++m) {
        auto maxDet = m_list[m];
        if (Eind.size() == 0) continue;
        
        // Concatenate first maxDet scores in each evalImg entry, -ve and sort w/indices
        std::vector<double> dtScores;
        for (size_t e = 0; e < Eind.size(); ++e) {
          auto score = dtScores_list[Eind[e]];
          for (size_t j = 0; j < std::min(score.size(), (size_t)maxDet); ++j) {
            dtScores.push_back(-score[j]);
          }
        }

        // get sorted indices of scores
        auto indices = stable_sort_indices(dtScores);
        std::vector<double> dtScoresSorted(dtScores.size());
        for (size_t j = 0; j < indices.size(); ++j) {
          dtScoresSorted[j] = -dtScores[indices[j]];
        }

        auto dtm = assemble_array<double>(Eind, dtMatches_list, T, maxDet, indices);
        auto dtIg = assemble_array<double>(Eind, dtIgnore_list, T, maxDet, indices);

        // gtIg = np.concatenate([e['gtIgnore'] for e in E])
        // npig = np.count_nonzero(gtIg==0 )
        int npig = 0;
        for (size_t e = 0; e < Eind.size(); ++e) {
          auto ignore = gtIgnore_list[Eind[e]];
          for (size_t j = 0; j < ignore.size(); ++j) {
            if(ignore[j] == 0) npig++;
          }
        }

        if (npig == 0) continue;

        int nrows = indices.size() ? dtm.size()/indices.size() : 0;
        std::vector<double> tp_sum(indices.size() * nrows);
        std::vector<double> fp_sum(indices.size() * nrows);
        for (int i = 0; i < nrows; ++i) {
          size_t tsum = 0, fsum = 0;
          for (size_t j = 0; j < indices.size(); ++j) {
            int index = i * indices.size() + j;
            tsum += (dtm[index]) && (!dtIg[index]);
            fsum += (!dtm[index]) && (!dtIg[index]);
            tp_sum[index] = tsum;
            fp_sum[index] = fsum;
          }
        }

        double eps = 2.220446049250313e-16; //std::numeric_limits<double>::epsilon();
        for (int t = 0; t < nrows; ++t) {
          // nd = len(tp)
          int nd = indices.size();
          std::vector<double> rc(indices.size());
          std::vector<double> pr(indices.size());
          for (size_t j = 0; j < indices.size(); ++j) {
            int index = t * indices.size() + j;
            // rc = tp / npig
            rc[j] = tp_sum[index] / npig;
            // pr = tp / (fp+tp+np.spacing(1))
            pr[j] = tp_sum[index] / (fp_sum[index]+tp_sum[index]+eps);
          }

          recall[t*K*A*M + k*A*M + a*M + m] = nd ? rc[indices.size()-1] : 0;

          std::vector<double> q(R);
          std::vector<double> ss(R);

          for (int i = 0; i < R; i++) {
            q[i] = 0;
            ss[i] = 0;
          }

          for (int i = nd-1; i > 0; --i) {
            if (pr[i] > pr[i-1]) {
              pr[i-1] = pr[i];
            }
          }

          std::vector<int> inds(recThrs.size());
          for (size_t i = 0; i < recThrs.size(); i++) {
            auto it = std::lower_bound(rc.begin(), rc.end(), recThrs[i]);
            inds[i] = it - rc.begin();
          }

          for (size_t ri = 0; ri < inds.size(); ri++) {
            size_t pi = inds[ri];
            if(pi>=pr.size())continue;
            q[ri] = pr[pi];
            ss[ri] = dtScoresSorted[pi];
          }

          for (size_t i = 0; i < inds.size(); i++) {
            // precision[t,:,k,a,m] = np.array(q)
            size_t index = t*R*K*A*M + i*K*A*M + k*A*M + a*M + m;
            precision[index] = q[i];
            scores[index] = ss[i];
          }
        }
      }
    }
  }

  py::array_t<double> precisionret = py::array_t<double>({T,R,K,A,M},{R*K*A*M*8,K*A*M*8,A*M*8,M*8,8},&precision[0]);
  py::array_t<double> recallret = py::array_t<double>({T,K,A,M},{K*A*M*8,A*M*8,M*8,8},&recall[0]);
  py::array_t<double> scoresret = py::array_t<double>({T,R,K,A,M},{R*K*A*M*8,K*A*M*8,A*M*8,M*8,8},&scores[0]);

  // clear arrays
  std::vector<int>().swap(evalImgs_list);
  std::vector<std::vector<int64_t>>().swap(gtIgnore_list);
  std::vector<std::vector<double>>().swap(dtIgnore_list);
  std::vector<std::vector<double>>().swap(dtMatches_list);
  std::vector<std::vector<double>>().swap(dtScores_list);
  std::unordered_map<size_t, data_struct>().swap(gts_map);
  std::unordered_map<size_t, data_struct>().swap(dts_map);

  return std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>>(precisionret,recallret,scoresret);
}

void bbIou(double *dt, double *gt, int m, int n, int *iscrowd, double *o ) {
  double h, w, i, u, ga, da; int g, d; int crowd;
  for( g=0; g<n; g++ ) {
    double* G=gt+g*4; ga=G[2]*G[3]; crowd=iscrowd!=NULL && iscrowd[g];
    for( d=0; d<m; d++ ) {
      double* D=dt+d*4; da=D[2]*D[3]; o[g*m+d]=0;
      w=fmin(D[2]+D[0],G[2]+G[0])-fmax(D[0],G[0]); if(w<=0) continue;
      h=fmin(D[3]+D[1],G[3]+G[1])-fmax(D[1],G[1]); if(h<=0) continue;
      i=w*h; u = crowd ? da : da+ga-i; o[g*m+d]=i/u;
    }
  }
}

py::array_t<double>
cpp_iou(py::array dt,
        py::array gt,
        std::vector<int> iscrowd) {
  auto dt_buf = dt.request();
  double *dt_ptr = (double *) dt_buf.ptr;
  auto gt_buf = gt.request();
  double *gt_ptr = (double *) gt_buf.ptr;
  int m = dt_buf.shape[0];//size();
  int n = gt_buf.shape[0];//size();
  std::vector<double> iou(m*n);
  if(m>0 && n>0) {
    bbIou(dt_ptr, gt_ptr, m, n, &iscrowd[0], &iou[0]);
  }
  return py::array_t<double>({m,n},{8,m*8},&iou[0]);
}

typedef struct { unsigned long h, w, m; unsigned int *cnts; } RLE;

void rleInit( RLE *R, unsigned long h, unsigned long w, unsigned long m, unsigned int *cnts ) {
  R->h=h; R->w=w; R->m=m; R->cnts=(m==0)?0:(unsigned int*)malloc(sizeof(unsigned int)*m);
  unsigned long j; if(cnts) for(j=0; j<m; j++) R->cnts[j]=cnts[j];
}

void rleFree( RLE *R ) {
  free(R->cnts); R->cnts=0;
}

void rleFrString( RLE *R, char *s, unsigned long h, unsigned long w ) {
  unsigned long m=0, p=0, k; long x; int more; unsigned int *cnts;
  while( s[m] ) m++; cnts=(unsigned int*)malloc(sizeof(unsigned int)*m); m=0;
  while( s[p] ) {
    x=0; k=0; more=1;
    while( more ) {
      char c=s[p]-48; x |= (c & 0x1f) << 5*k;
      more = c & 0x20; p++; k++;
      if(!more && (c & 0x10)) x |= -1 << 5*k;
    }
    if(m>2) x+=(long) cnts[m-2]; cnts[m++]=(unsigned int) x;
  }
  rleInit(R,h,w,m,cnts); free(cnts);
}

unsigned int umin( unsigned int a, unsigned int b ) { return (a<b) ? a : b; }
unsigned int umax( unsigned int a, unsigned int b ) { return (a>b) ? a : b; }

void rleArea( const RLE *R, unsigned long n, unsigned int *a ) {
  unsigned long i, j; for( i=0; i<n; i++ ) {
    a[i]=0; for( j=1; j<R[i].m; j+=2 ) a[i]+=R[i].cnts[j]; }
}

void rleToBbox( const RLE *R, double* bb, unsigned long n ) {
  unsigned long i; for( i=0; i<n; i++ ) {
    unsigned int h, w, x, y, xs, ys, xe, ye, xp=0, cc, t; unsigned long j, m;
    h=(unsigned int)R[i].h; w=(unsigned int)R[i].w; m=R[i].m;
    m=((unsigned long)(m/2))*2; xs=w; ys=h; xe=ye=0; cc=0;
    if(m==0) { bb[4*i+0]=bb[4*i+1]=bb[4*i+2]=bb[4*i+3]=0; continue; }
    for( j=0; j<m; j++ ) {
      cc+=R[i].cnts[j]; t=cc-j%2; y=t%h; x=(t-y)/h;
      if(j%2==0) xp=x; else if(xp<x) { ys=0; ye=h-1; }
      xs=umin(xs,x); xe=umax(xe,x); ys=umin(ys,y); ye=umax(ye,y);
    }
    bb[4*i+0]=xs; bb[4*i+2]=xe-xs+1;
    bb[4*i+1]=ys; bb[4*i+3]=ye-ys+1;
  }
}

void rleIou( RLE *dt, RLE *gt, int m, int n, int *iscrowd, double *o ) {
  int g, d; double *db, *gb; int crowd;
  db=(double*)malloc(sizeof(double)*m*4); rleToBbox(dt,db,m);
  gb=(double*)malloc(sizeof(double)*n*4); rleToBbox(gt,gb,n);
  bbIou(db,gb,m,n,iscrowd,o); free(db); free(gb);
  for( g=0; g<n; g++ ) for( d=0; d<m; d++ ) if(o[g*m+d]>0) {
    crowd=iscrowd!=NULL && iscrowd[g];
    if(dt[d].h!=gt[g].h || dt[d].w!=gt[g].w) { o[g*m+d]=-1; continue; }
    unsigned long ka, kb, a, b; uint c, ca, cb, ct, i, u; int va, vb;
    ca=dt[d].cnts[0]; ka=dt[d].m; va=vb=0;
    cb=gt[g].cnts[0]; kb=gt[g].m; a=b=1; i=u=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); if(va||vb) { u+=c; if(va&&vb) i+=c; } ct=0;
      ca-=c; if(!ca && a<ka) { ca=dt[d].cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<kb) { cb=gt[g].cnts[b++]; vb=!vb; } ct+=cb;
    }
    if(i==0) u=1; else if(crowd) rleArea(dt+d,1,&u);
    o[g*m+d] = (double)i/(double)u;
  }
}

py::dict
cpp_compute_iou(std::vector<int> imgIds,
                std::vector<int> catIds,
                py::dict gts,
                py::dict dts,
                std::string iouType,
                int maxDet, int useCats) {
  assert(useCats > 0);

  py::dict list;

  for(size_t i = 0; i < imgIds.size(); i++) {
  for(size_t c = 0; c < catIds.size(); c++) {
    int catId = catIds[c];
    int imgId = imgIds[i];

    auto gtsm = &gts_map[key(imgId,catId)];
    auto dtsm = &dts_map[key(imgId,catId)];
    auto G = gtsm->id.size();
    auto D = dtsm->id.size();

    py::tuple k = py::make_tuple(imgId,catId);

    if((G==0) && (D==0)) {
      list[k] = py::array_t<double>();
      continue;
    }

    std::vector<float> scores;
    for (size_t i = 0; i < D; i++) {
      auto score = -dtsm->score[i];
      scores.push_back(score);
    }
    auto inds = sort_indices(scores);

    assert(iouType=="bbox"||iouType=="segm");

    if (iouType == "bbox") {
      std::vector<double> g;
      for (size_t i = 0; i < G; i++) {
        auto arr = gtsm->bbox[i];
        for (size_t j = 0; j < arr.size(); j++) {
          g.push_back((double)arr[j]);
        }
      }
      std::vector<double> d;
      for (size_t i = 0; i < std::min(D,(size_t)maxDet); i++) {
        auto arr = dtsm->bbox[inds[i]];
        for (size_t j = 0; j < arr.size(); j++) {
          d.push_back((double)arr[j]);
        }
      }
      // compute iou between each dt and gt region
      // iscrowd = [int(o['iscrowd']) for o in gt]
      std::vector<int> iscrowd(G);
      for (size_t i = 0; i < G; i++) {
        iscrowd[i] = gtsm->iscrowd[i];
      }
      int m = std::min(D,(size_t)maxDet);
      int n = G;
      if(m==0 || n==0) {
        list[k] = py::array_t<double>();
        continue;
      }
      std::vector<double> iou(m*n);
      // internal conversion from compressed RLE format to Python RLEs object
      //if (iouType == "bbox")
      bbIou(&d[0], &g[0], m, n, &iscrowd[0], &iou[0]);
      //rleIou(&dt[0],&gt[0],m,n,&iscrowd[0], double *o )
      list[k] = py::array_t<double>({m,n},{8,m*8},&iou[0]);
    }
    else {
      std::vector<RLE> g(G);
      for (size_t i = 0; i < G; i++) {
        auto size = gtsm->segm_size[i];
        auto str = gtsm->segm_counts[i];
        char *val = new char[str.length() + 1];
        strcpy(val, str.c_str());
        rleFrString(&g[i],val,size[0],size[1]);
        delete [] val;
      }
      std::vector<RLE> d(std::min(D,(size_t)maxDet));
      for (size_t i = 0; i < std::min(D,(size_t)maxDet); i++) {
        auto size = dtsm->segm_size[i];
        auto str = dtsm->segm_counts[inds[i]];
        char *val = new char[str.length() + 1];
        strcpy(val, str.c_str());
        rleFrString(&d[i],val,size[0],size[1]);
        delete [] val;
      }
      std::vector<int> iscrowd(G);
      for (size_t i = 0; i < G; i++) {
        iscrowd[i] = gtsm->iscrowd[i];
      }
      int m = std::min(D,(size_t)maxDet);
      int n = G;
      if(m==0 || n==0) {
        list[k] = py::array_t<double>();
        continue;
      }
      std::vector<double> iou(m*n);
      // internal conversion from compressed RLE format to Python RLEs object
      //if (iouType == "bbox")
      //bbIou(&d[0], &g[0], m, n, &iscrowd[0], &iou[0]);
      rleIou(&d[0], &g[0], m, n, &iscrowd[0], &iou[0]);
      list[k] = py::array_t<double>({m,n},{8,m*8},&iou[0]);
    }
  }}
  return list;
}

void cpp_prepare(py::list gts,
                 py::list dts,
                 std::string iouType) {
  // create dictionary
  for (size_t i = 0; i < gts.size(); i++) {
    auto imgId = py::cast<int>(gts[i]["image_id"]);
    auto catId = py::cast<int>(gts[i]["category_id"]);
    auto k = key(imgId,catId);

    auto area = py::cast<float>(gts[i]["area"]);
    gts_map[k].area.push_back(area);

    auto iscrowd = py::cast<int>(gts[i]["iscrowd"]);
    gts_map[k].iscrowd.push_back(iscrowd);

    auto bbox = py::cast<std::vector<float>>(gts[i]["bbox"]);
    gts_map[k].bbox.push_back(bbox);

    auto ignore = py::cast<int>(gts[i]["ignore"]);
    gts_map[k].ignore.push_back(ignore);

    if(iouType == "segm") {
      auto segm_size = py::cast<std::vector<int>>(gts[i]["segmentation"]["size"]);
      auto segm_counts = py::cast<std::string>(gts[i]["segmentation"]["counts"]);
      gts_map[k].segm_size.push_back(segm_size);
      gts_map[k].segm_counts.push_back(segm_counts);
    }

    auto id = py::cast<float>(gts[i]["id"]);
    gts_map[k].id.push_back(id);
  }

  for (size_t i = 0; i < dts.size(); i++) {
    auto imgId = py::cast<int>(dts[i]["image_id"]);
    auto catId = py::cast<int>(dts[i]["category_id"]);
    auto k = key(imgId,catId);

    auto area = py::cast<float>(dts[i]["area"]);
    dts_map[k].area.push_back(area);

    auto iscrowd = py::cast<int>(dts[i]["iscrowd"]);
    dts_map[k].iscrowd.push_back(iscrowd);

    auto bbox = py::cast<std::vector<float>>(dts[i]["bbox"]);
    dts_map[k].bbox.push_back(bbox);

    auto score = py::cast<float>(dts[i]["score"]);
    dts_map[k].score.push_back(score);

    if(iouType == "segm") {
      auto segm_size = py::cast<std::vector<int>>(dts[i]["segmentation"]["size"]);
      auto segm_counts = py::cast<std::string>(dts[i]["segmentation"]["counts"]);
      dts_map[k].segm_size.push_back(segm_size);
      dts_map[k].segm_counts.push_back(segm_counts);
    }

    auto id = py::cast<float>(dts[i]["id"]);
    dts_map[k].id.push_back(id);
  }
}

PYBIND11_MODULE(ext, m) {
    m.doc() = "pybind11 pycocotools plugin";
    m.def("cpp_evaluate_img", &cpp_evaluate_img, "");
    m.def("cpp_accumulate", &cpp_accumulate, "");
    m.def("cpp_compute_iou", &cpp_compute_iou, "");
    m.def("cpp_prepare", &cpp_prepare, "");
}
