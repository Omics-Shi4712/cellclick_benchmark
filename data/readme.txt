marker_list
 celltypegpt/
   celltypegpt.tsv (derived from 41592_2024_2235_MOESM3_ESM.xlsx, supplementary table 4; https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41592-024-02235-4/MediaObjects/41592_2024_2235_MOESM3_ESM.xlsx; ref: Assessing GPT-4 for cell type annotation in single-cell RNA-seq analysis)
   41592_2024_2235_MOESM3_ESM.xlsx (source workbook)
   celltype.config.json (adapter configuration)
 cissia/
   cassia.tsv (derived from Supplementary Data 5.xlsx; https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-67084-x/MediaObjects/41467_2025_67084_MOESM3_ESM.zip; ref: CASSIA: a multi-agent large language model for automated and interpretable cell annotation)
   Supplementary Data 5.xlsx (source workbook)
   cassia.config.json (adapter configuration)

expression_data
  ###############
  GTEx (GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad from https://gtexportal.org/home/downloads/adult-gtex/single_cell; ref： Single-nucleus cross-tissue molecular reference maps toward understanding disease gene function)
  .X: nor_log1p matrix
  .layers["counts"] raw_counts matrix
  .obs["Tissue"]:  (celltypegpt_mark "tissue"， cassia "Tissue")
  .obs["Broad cell type"]: ( celltypegpt_mark "mannual annotation", cassia "True Cell Type")

  ######  与celltypegpt一致性检验
    数据源                       tissue 数    cell type 数    实际出现的 tissue × cell type 组合
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   h5ad（原始组织层级）                 8              44                             114 / 352
  ───────────────────────────  ───────────  ──────────────  ────────────────────────────────────
   h5ad（合并 Esophagus 后）            7              44                             104 / 308
  ───────────────────────────  ───────────  ──────────────  ────────────────────────────────────
   marker TSV（GTEx）                   7              41                              99 / 287
  
     Tissue               h5ad cells    h5ad types    marker types / pairs    重叠类型    重叠细胞
  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━
   Breast                    9,770             8                       8           8       9,770
  ───────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   Esophagus（合并）        60,233            21                      17          17      46,134
  ───────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   Heart                    36,574            13                      13          13      36,574
  ───────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   Lung                     35,284            15                      15          15      35,284
  ───────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   Prostate                 31,061            16                      16          16      31,061
  ───────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   Skeletal muscle          30,877            14                      14          14      30,877
  ───────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   Skin                      5,327            17                      16          16       4,875
  
  * marker文件将 Esophagus muscularis 和 Esophagus mucosa 合并为Esophagus
  * 未涵盖细胞类型
    - Esophagus：Adipocyte（175 cells）、ICCs（237）、Myocyte (smooth muscle)（13,662）、Neuronal（25）
    - Skin：Unknown（452）


  ###### 与 CASSIA 一致性检验
  数据源                  tissue 数    cell type 数    实际出现的 tissue × cell type 组合
  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   h5ad（原始组织层级）            8              44                             114 / 352
  ──────────────────────  ───────────  ──────────────  ────────────────────────────────────
   CASSIA TSV（GTEX）              7              42                             105 / 294

   Tissue                 h5ad cells    h5ad types    CASSIA types / pairs    重叠类型    重叠细胞
  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━
   skeletalmuscle             30,877            14                      14          14      30,877
  ─────────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   heart                      36,574            13                      13          13      36,574
  ─────────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   esophagusmucosa            26,060            17                      17          17      26,060
  ─────────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   esophagusmuscularis        34,173            14                      14          14      34,173
  ─────────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   lung                       35,284            15                      15          15      35,284
  ─────────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   skin                        5,327            17                      16          16       4,875
  ─────────────────────  ────────────  ────────────  ──────────────────────  ──────────  ──────────
   prostate                   31,061            16                      16          16      31,061

   * marker文件未覆盖h5ad的组织是 Breast
   * 未涵盖细胞类型
     - Breast 的全部 8 类：Adipocyte、Endothelial cell (lymphatic)、Endothelial cell (vascular)、Epithelial cell (luminal)、Fibroblast、Immune (DC/macrophage)、Myoepithelial (basal)、Pericyte/SMC（共 9,770 cells）。
     - Skin：Unknown（452 cells）。
  ###############

  ###############
  TS_v2  (1c88f927-bcbb-4bb1-9881-281842945a2d.h5ad from https://cellxgene.cziscience.com/e/53d208b0-2cfd-4366-9866-c3c6114081bc.cxg/; ref： Tabula Sapiens 2.0: A comprehensive transcriptomic atlas of human cell types)
  .layers["decontXcounts"] raw_counts matrix
  .obs["tissue_in_publication"]
  .obs["free_annotation"]

  ** 稀有细胞一览
   Tissue            细胞数    cell type 数    仅 1 个细胞    ≤2 个细胞    unknown
  ━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━
   Testis             7,513              15              4            5          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Fat               94,415              28              3            4          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Salivary_Gland    39,821              31              3            4          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Ovary             48,951              27              2            4          9
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Trachea           22,671              28              2            3          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Uterus            22,029              28              2            4          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Skin              17,786              25              2            2          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Pancreas          14,140              23              2            5          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Blood             85,233              22              1            1          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Spleen            70,448              26              1            3          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Muscle            46,772              32              1            2          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Thymus            42,729              33              1            2          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Vasculature       42,650              25              1            1          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Eye               34,273              39              1            3          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Heart             25,832              22              1            1          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   Ear                3,055              16              1            4          0
  ────────────────  ────────  ──────────────  ─────────────  ───────────  ─────────
   其余 12 个组织         —               —              0       0 或 1          0


  MCA (5435866.zip from https://figshare.com/articles/dataset/MCA_DGE_Data/5435866?utm_source=chatgpt.com&file=22008951; ref： Mapping the Mouse Cell Atlas by Microwell-Seq)
  .X raw_counts matrix
  .obs["tissue_in_publication"] -> MCA_CellAssignments.csv
  .obs["cell_type"]

  ~6w cell subset was used based on 
  /gpfs/flash/home/sl/projects/scRNA/AI_CTA_benchmark/data/expression_data/MCA/5435866/MCA_Figure2-batch-removed.txt.tar.gz
  /gpfs/flash/home/sl/projects/scRNA/AI_CTA_benchmark/data/expression_data/MCA/5435866/MCA_Figure2_Cell.Info.xlsx
  .X
  .obs["tissue"]
  .obs["cell_type"]

  ###############
  cancer( 1b76227b-c731-4807-9487-ad5e4d24e0d0.h5ad from https://cellxgene.cziscience.com/collections/3f7c572c-cd73-4b51-a313-207c7f20f188; ref： Single-cell resolution characterization of myeloid-derived cell states with implication in cancer outcome)
  .X raw_counts matrix
  .obs["tissue"]
  .obs["cell_type"]

  ###############
  TS_v1 (1c88f927-bcbb-4bb1-9881-281842945a2d.h5ad from https://cells.ucsc.edu/?ds=tabula-sapiens&utm_source=chatgpt.com; ref： Tabula Sapiens 2.0: The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans1）)