"""
Өгөгдлийн боловсруулалтын модуль

Зээлийн эрсдэлийн өгөгдлийг цэвэрлэж,
боловсруулж, загварт бэлтгэх функцууд
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(filepath: str) -> pd.DataFrame:
    """
    CSV файлаас өгөгдөл уншиж авах

    Parameters:
    -----------
    filepath : str
        CSV файлын зам

    Returns:
    --------
    pd.DataFrame
        Уншсан өгөгдөл
    """
    df = pd.read_csv(filepath)
    print(f"Өгөгдөл уншигдлаа: {df.shape[0]} мөр, {df.shape[1]} багана")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дутуу утгуудыг шалгах

    Parameters:
    -----------
    df : pd.DataFrame
        Шалгах өгөгдөл

    Returns:
    --------
    pd.DataFrame
        Дутуу утгын мэдээлэл
    """
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100

    missing_df = pd.DataFrame({
        'Багана': missing_data.index,
        'Дутуу тоо': missing_data.values,
        'Хувь (%)': np.round(missing_pct.values, 2)
    })

    missing_df = missing_df[missing_df['Дутуу тоо'] > 0].sort_values(
        'Дутуу тоо', ascending=False
    )

    return missing_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дутуу утгуудыг бөглөх

    Тоон багануудад median-аар,
    Ангиллын багануудад mode-оор бөглөнө

    Parameters:
    -----------
    df : pd.DataFrame
        Өгөгдөл

    Returns:
    --------
    pd.DataFrame
        Дутуу утга бөглөгдсөн өгөгдөл
    """
    df_filled = df.copy()

    # Тоон багануудад median-аар бөглөх
    num_cols = df_filled.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df_filled[num_cols] = imputer_num.fit_transform(df_filled[num_cols])

    # Ангиллын багануудад mode-оор бөглөх
    cat_cols = df_filled.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_filled[cat_cols] = imputer_cat.fit_transform(df_filled[cat_cols])

    print(f"Дутуу утга бөглөгдлөө. Үлдсэн дутуу утга: {df_filled.isnull().sum().sum()}")

    return df_filled


def encode_categorical(X: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    Ангиллын хувьсагчдыг one-hot encoding хийх

    Parameters:
    -----------
    X : pd.DataFrame
        Feature-ууд
    drop_first : bool
        Анхны dummy хувьсагчийг устгах эсэх

    Returns:
    --------
    pd.DataFrame
        Encoding хийгдсэн өгөгдөл
    """
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    if len(cat_cols) > 0:
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=drop_first)
        print(f"One-hot encoding: {len(cat_cols)} багана -> {X_encoded.shape[1]} багана")
        return X_encoded
    else:
        print("Ангиллын багана олдсонгүй")
        return X.copy()


def split_features_target(
    df: pd.DataFrame,
    target_col: str = 'loan_status'
) -> tuple:
    """
    Өгөгдлийг features болон target-д салгах

    Parameters:
    -----------
    df : pd.DataFrame
        Өгөгдөл
    target_col : str
        Зорилтот баганын нэр

    Returns:
    --------
    tuple
        (X, y) - Features болон target
    """
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' багана олдсонгүй")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    print(f"Features: {X.shape[1]} багана")
    print(f"Target: {target_col}")

    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> tuple:
    """
    Train болон test багц болгон хуваах

    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Test багцын хувь
    random_state : int
        Random seed
    stratify : bool
        Stratified split хийх эсэх

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    print(f"Train багц: {X_train.shape[0]} өгөгдөл")
    print(f"Test багц: {X_test.shape[0]} өгөгдөл")

    if stratify:
        print(f"Train default rate: {y_train.mean()*100:.2f}%")
        print(f"Test default rate: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str = 'standard'
) -> tuple:
    """
    Features-ийг стандартчлах

    Parameters:
    -----------
    X_train : np.ndarray
        Train багц
    X_test : np.ndarray
        Test багц
    method : str
        Стандартчлах арга ('standard')

    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Дэмжигдэхгүй арга: {method}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Стандартчлал амжилттай ({method})")

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    filepath: str,
    target_col: str = 'loan_status',
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Бүтэн боловсруулалтын pipeline

    Parameters:
    -----------
    filepath : str
        Өгөгдлийн файлын зам
    target_col : str
        Зорилтот баганын нэр
    test_size : float
        Test багцын хувь
    random_state : int
        Random seed

    Returns:
    --------
    dict
        Боловсруулсан өгөгдөл агуулсан dictionary:
        - X_train_scaled
        - X_test_scaled
        - y_train
        - y_test
        - scaler
        - feature_names
    """
    # 1. Өгөгдөл уншиж авах
    df = load_data(filepath)

    # 2. Дутуу утга шалгах
    missing_df = check_missing_values(df)
    if len(missing_df) > 0:
        print("\nДутуу утгатай баганууд:")
        print(missing_df)

    # 3. Дутуу утга бөглөх
    df = fill_missing_values(df)

    # 4. Features болон target салгах
    X, y = split_features_target(df, target_col)

    # 5. One-hot encoding
    X_encoded = encode_categorical(X, drop_first=True)

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X_encoded, y,
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )

    # 7. Стандартчлал
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train.values, X_test.values
    )

    # Үр дүн буцаах
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X_encoded.columns.tolist()
    }


if __name__ == "__main__":
    # Жишээ ашиглалт
    data = preprocess_pipeline(
        filepath='data/credit_risk_dataset.csv',
        target_col='loan_status',
        test_size=0.2,
        random_state=42
    )

    print("\n=== Боловсруулалт дууссан ===")
    print(f"Train shape: {data['X_train_scaled'].shape}")
    print(f"Test shape: {data['X_test_scaled'].shape}")
    print(f"Feature тоо: {len(data['feature_names'])}")
