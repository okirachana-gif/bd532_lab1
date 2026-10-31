"""ส่วนประกอบของ ML pipeline ที่ใช้ร่วมกันทั้งตอนเทรน (Lab 1.1) และตอนให้บริการ API (Lab 1.2)"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# คอลัมน์ตัวเลข (รวมคอลัมน์ข้อความที่มีลำดับ ซึ่งจะถูกแปลงเป็นตัวเลขก่อน)
NUM_COLS = ["city_development_index", "relevent_experience", "experience", "last_new_job", "training_hours"]
# คอลัมน์ประเภท (categorical) ที่ไม่มีลำดับ
CAT_COLS = ["gender", "enrolled_university", "education_level", "major_discipline", "company_size", "company_type"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "target"

# ตารางแปลงค่าข้อความที่มีลำดับเป็นตัวเลข
ORDINAL_MAPS = {
    "relevent_experience": {"Has relevent experience": 1, "No relevent experience": 0},
    "experience": {"<1": 0, **{str(i): i for i in range(1, 21)}, ">20": 21},
    "last_new_job": {"never": 0, "1": 1, "2": 2, "3": 3, "4": 4, ">4": 5},
}


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """Custom transformer: แปลงคอลัมน์ข้อความที่มีลำดับ (เช่น '<1', '5', '>20') เป็นตัวเลข

    ค่าที่ไม่อยู่ในตารางหรือค่าว่างจะกลายเป็น NaN แล้วให้ SimpleImputer ในขั้นถัดไปเติมค่า
    """

    def fit(self, X, y=None):
        return self  # ไม่ต้องเรียนรู้อะไรจากข้อมูล

    def transform(self, X):
        X = X.copy()
        for col, mapping in ORDINAL_MAPS.items():
            X[col] = X[col].map(mapping).astype(float)
        return X


def build_pipeline(model=None):
    """สร้าง pipeline: แปลงค่าที่มีลำดับ → เตรียมคอลัมน์ตัวเลขและประเภทแยกกัน → โมเดล"""
    num_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", MinMaxScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    col_trans = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
        ],
        remainder="drop",
    )
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=0)
    return Pipeline(steps=[
        ("encode", OrdinalMapper()),
        ("col_trans", col_trans),
        ("model", model),
    ])
