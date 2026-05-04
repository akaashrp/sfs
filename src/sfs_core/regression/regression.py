import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def _normalize_dir(path_str):
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        candidate = Path("/") / path
        if candidate.exists():
            path = candidate
        else:
            raise ValueError(f"Expected absolute path for directory, got '{path_str}'.")
    return str(path)

def regress(filename, path, cols = ['ts','engine','prefill','prefill_sq_sum','decode','decode_sq_sum','total','sched','exec','interval','num_seqs','sum_tokens','sum_sq_tokens','avg_tokens','max_tokens']):
    path = _normalize_dir(path)
    df = pd.read_csv(path, header=0, names=cols)
    
    prefill_tokens = df['prefill']
    prefill_tokens_sq = df['prefill'] ** 2
    prefill_tokens_sq_sum = df['prefill_sq_sum']
    decode_tokens = df['decode']
    decode_tokens_sq = df['decode'] ** 2
    sum_tokens = df['sum_tokens']
    sum_sq_tokens = df['sum_sq_tokens']
    avg_tokens = df['avg_tokens']
    max_tokens = df['max_tokens']
    y = df['exec']
    
    # df['interval'] = df['interval'].shift(-1).fillna(0)

    feature_names = ["p", "p_sq", "p_sq_sum", "d", "pd", "s", "s_sq", "a"]
    feature_arrays = [prefill_tokens, prefill_tokens_sq, prefill_tokens_sq_sum, decode_tokens, prefill_tokens * decode_tokens, sum_tokens, sum_sq_tokens, avg_tokens]
    # feature_names = ["p", "d"]
    # feature_arrays = [prefill_tokens, decode_tokens]
    X = np.column_stack(feature_arrays)

    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(prefill_tokens, decode_tokens, y, marker="o", alpha=0.6, label="Actual")
    ax.scatter(prefill_tokens, decode_tokens, y_pred, marker="x", color="red", alpha=0.6, label="Predicted")
    ax.set_xlabel("Batch Num Prefill Tokens")
    ax.set_ylabel("Batch Num Decode Tokens")
    ax.set_zlabel("Batch Execution Time")
    ax.set_title("Batch Execution Time vs Prefill/Decode Tokens")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()

    r2 = reg.score(X, y)
    print(f"R^2 value of regression: {r2}")
    out_str = "Regression coefficients:\n"
    for name, coef in zip(feature_names, reg.coef_):
        out_str += f"beta_{name}={coef}\n"
    out_str += f"intercept={reg.intercept_}\n"
    print(out_str)
    
    with open(f"{filename}_regression_coefficients.txt", "w") as f:
        f.write(out_str)
        f.write(f"sum(residuals): {residuals.sum()}, mean: {residuals.mean()}, std: {residuals.std()}\n")
        f.write(f"R^2 value: {r2}\n")

    # for feature_name, feature_values in zip(
    #     feature_names,
    #     feature_arrays
    # ):
    #     plt.figure(figsize=(6, 4))
    #     plt.scatter(feature_values, residuals, alpha=0.6)
    #     plt.axhline(0.0, color="red", linestyle="--")
    #     plt.xlabel(feature_name)
    #     plt.ylabel("Residual")
    #     plt.title(f"Residuals vs {feature_name}")
    #     plt.tight_layout()
    #     plt.savefig(f"{filename}_residuals_vs_{feature_name}.png")
    #     plt.clf()
    #     plt.close()

if __name__ == "__main__":
    path = Path('/ocean/projects/cis250162p/aparthas/sfs/experiments/batches_qwen3_8b_new_data.csv')
    regress('regression_qwen3_8b_new_data.png', path)
    path = Path('/ocean/projects/cis250162p/aparthas/sfs/experiments/batches_qwen3_32b_new_data.csv')
    regress('regression_qwen3_32b_new_data.png', path)
    path = Path('/ocean/projects/cis250162p/aparthas/sfs/experiments/batches_qwen3_0.6b_new_data.csv')
    regress('regression_qwen3_0.6b_new_data.png', path)

# try for non power of 2 number of sequences
# try different sequences (make sure every sequence is different) - maybe randomly generate 6000 numbers
# determine if scheduling simulator is blocking
