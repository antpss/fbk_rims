"""Train-fitted encodings shared by timing training and inference."""
import hashlib
import pandas as pd


def fit_encoding(frame, columns, cfg, engine):
    declared = {a['feature_name'] for a in cfg.get('features', {}).get('case_attributes', [])
                if a.get('type') != 'continuous'}
    categories = {}
    for col in columns:
        if col in declared or col == 'Resource_ID' or not pd.api.types.is_numeric_dtype(frame[col]):
            categories[col] = sorted(frame[col].dropna().astype(str).unique().tolist())
    outputs = {}
    if engine != 'xgboost':
        for col, values in categories.items():
            ident = hashlib.sha256(col.encode()).hexdigest()
            outputs[col] = [f'__category_{ident}_{i}' for i in range(len(values) + 1)]
        generated = [name for names in outputs.values() for name in names]
        if set(generated).intersection(columns):
            raise ValueError('Input feature name collides with generated category encoding')
    return {'categories': categories, 'outputs': outputs, 'engine': engine}


def encode_frame(frame, schema):
    result = frame.copy()
    for col, values in schema['categories'].items():
        series = frame[col].astype('string')
        if schema['engine'] == 'xgboost':
            result[col] = pd.Categorical(series, categories=values)
        else:
            result = result.drop(columns=col)
            for value, name in zip(values, schema['outputs'][col]):
                result[name] = series.eq(value).fillna(False).astype(float)
            result[schema['outputs'][col][-1]] = (~series.isin(values)).astype(float)
    return result


def encode_values(values, schema):
    """XGB uses category codes; neural models use one-hot plus unknown indicators."""
    result = dict(values)
    for col, categories in schema.get('categories', {}).items():
        value = values.get(col)
        value = None if value is None or pd.isna(value) else str(value)
        if schema['engine'] == 'xgboost':
            result[col] = float(categories.index(value)) if value in categories else float('nan')
        else:
            result.pop(col, None)
            for category, name in zip(categories, schema['outputs'][col]):
                result[name] = float(value == category)
            result[schema['outputs'][col][-1]] = float(value not in categories)
    return result
