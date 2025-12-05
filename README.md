# Зээлийн эрсдэлийн үнэлгээ - Naive Bayes ангилагч

Энэхүү төсөл нь банкны зээлийн эрсдэлийг урьдчилан таамаглах зорилгоор Naive Bayes ангилагчийг ашигласан судалгаа юм.

## Агуулга

- [Тойм](#тойм)
- [Өгөгдөл](#өгөгдөл)
- [Арга зүй](#арга-зүй)
- [Төслийн бүтэц](#төслийн-бүтэц)
- [Суулгалт](#суулгалт)
- [Ашиглалт](#ашиглалт)
- [Үр дүн](#үр-дүн)
- [Багийн гишүүд](#багийн-гишүүд)

## Тойм

Зээлийн эрсдэлийн удирдлага нь банк санхүүгийн байгууллагын хамгийн чухал асуудлуудын нэг юм. Энэхүү судалгаагаар:

- **Зорилго**: Зээлдэгчийн төлбөрийн чадварыг (default эсэх) урьдчилан таамаглах
- **Загвар**: Naive Bayes (Gaussian)
- **Үнэлгээ**: Accuracy, Precision, Recall, F1-Score, AUC, ROC Curve
- **Хэрэглэгч**: Python 3.9+

## Өгөгдөл

**Эх сурвалж**: Kaggle - [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

**Хэмжээ**: 32,581 бичлэг, 12 хувьсагч

**Хувьсагчид**:
- `person_age`: Зээлдэгчийн нас
- `person_income`: Жилийн орлого
- `person_home_ownership`: Орон сууцны эзэмшил
- `person_emp_length`: Ажилласан жил
- `loan_intent`: Зээлийн зориулалт
- `loan_grade`: Зээлийн зэрэглэл
- `loan_amnt`: Зээлийн хэмжээ
- `loan_int_rate`: Зээлийн хүү
- `loan_status`: **Зорилтот хувьсагч** (0=төлсөн, 1=default)
- `loan_percent_income`: Зээлийн орлогод эзлэх хувь
- `cb_person_default_on_file`: Өмнө default байсан эсэх
- `cb_person_cred_hist_length`: Зээлийн түүхийн урт

## Арга зүй

### 1. Өгөгдөл боловсруулалт
- Дутуу утга бөглөх (median/mode)
- One-hot encoding (ангиллын хувьсагчид)
- Train-Test хуваалт (80-20, stratified)
- Стандартчлал (StandardScaler)

### 2. Naive Bayes загвар

**Gaussian Naive Bayes**
- Bayesian-ы таамаглалд суурилсан
- Feature-үүд бие биенээсээ хамааралгүй гэсэн таамаглал
- Хурдан, энгийн алгоритм
- Өгөгдлийг тасралтгүй хэмжээст (continuous) гэж үзнэ
- Бага өгөгдөлтэй ажиллах чадвартай

### 3. Үнэлгээ
- Accuracy, Precision, Recall, F1-Score
- ROC Curve, AUC
- Confusion Matrix

## Төслийн бүтэц

```
MSproject/
│
├── data/
│   └── credit_risk_dataset.csv       # Өгөгдөл
│
├── src/
│   ├── preprocessing.py              # Өгөгдөл боловсруулалт
│   └── models.py                     # Загваруудын сургалт ба үнэлгээ
│
├── notebook/
│   └── analysis.ipynb                # Naive Bayes шинжилгээ
│
├── outputs/                          # Үр дүнгийн зураг, хүснэгт
│
├── _files/                           # QMD-ээс үүссэн файлууд
│
├── report.qmd                        # Үндсэн тайлан (Quarto)
├── references.bib                    # Эх сурвалжийн жагсаалт
├── ieee.csl                          # Citation style
├── README.md                         # Энэ файл
└── .gitignore

```

## Суулгалт

### 1. Repository татаж авах

```bash
git clone https://github.com/yourusername/MSproject.git
cd MSproject
```

### 2. Python орчин үүсгэх

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Шаардлагатай сангууд суулгах

```bash
pip install -r requirements.txt
```

**Үндсэн сангууд**:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

### 4. R орчин (тайлан үүсгэхэд)

```bash
# Quarto суулгах
# https://quarto.org/docs/get-started/

# R packages
install.packages(c("knitr", "rmarkdown"))
```

## Ашиглалт

### 1. Jupyter Notebook ажиллуулах

```bash
jupyter notebook notebook/analysis.ipynb
```

### 2. Python скрипт ажиллуулах

```bash
# Өгөгдөл боловсруулалт
python src/preprocessing.py

# Загваруудын ажиллуулалт (ирээдүйд)
python src/models.py
```

### 3. Тайлан үүсгэх (PDF)

```bash
quarto render report.qmd
```

## Үр дүн

Naive Bayes загварын гүйцэтгэл (Test багц дээр):

| Үзүүлэлт | Утга |
|----------|------|
| Accuracy | ~0.83 |
| Precision | ~0.65 |
| Recall | ~0.43 |
| F1-Score | ~0.52 |
| AUC | ~0.86 |

**Гол дүгнэлт**:
1. Загвар сайн ялгах чадвартай (AUC > 0.85)
2. Хурдан, энгийн алгоритм бөгөөд тайлбарлахад хялбар
3. Precision өндөр - default таамагласан тохиолдолд итгэлтэй
4. Recall харьцангуй доогуур - зарим default тохиолдлыг алдаж байна
5. Class imbalance нөлөөлж байгаа тул цаашид SMOTE эсвэл class weights ашиглах хэрэгтэй

## Багийн гишүүд

**Д.Эрдэнэтуяа**
- Өгөгдлийн цуглуулалт ба боловсруулалт
- Naive Bayes загварын хэрэгжүүлэлт
- Тайлангийн бүтэц, дизайн ба форматчлал

**Б.Байрбилэг**
- Танин мэдэхүйн өгөгдлийн шинжилгээ (EDA)
- Үр дүнгийн үнэлгээ ба дүрслэл
- Дүгнэлт ба цаашдын сайжруулалтын санал

## Эх сурвалж

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
2. Lantz, B. (2019). *Machine Learning with R* (3rd ed.). Packt Publishing.
3. Kaggle Credit Risk Dataset: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

## License

Энэхүү төсөл нь сургалтын зориулалттай бөгөөд MIT License-тэй.

---

**Төслийн хугацаа**: 2025 оны 12-р сар
**Сургууль**: Монгол Улсын Их Сургууль
**Хичээл**: Магадлал, Статистик
