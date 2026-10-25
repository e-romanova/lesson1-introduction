"""Решение задания 2: autograd, MSE и цепное правило."""
import math
import torch


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Средняя квадратичная ошибка; формы входов должны совпадать."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Формы должны совпадать: {y_pred.shape} != {y_true.shape}")
    if not y_pred.is_floating_point() or not y_true.is_floating_point():
        raise TypeError("Для MSE нужны тензоры с плавающей точкой")
    return ((y_pred - y_true) ** 2).mean()


def main() -> None:
    # 2.1 f(x,y,z) = x^2 + y^2 + z^2 + 2xyz
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)
    f = x**2 + y**2 + z**2 + 2*x*y*z
    f.backward()
    print("=== 2.1 Градиенты простой функции ===")
    print(f"f={f.item():.4f}; df/dx={x.grad.item():.4f}; df/dy={y.grad.item():.4f}; df/dz={z.grad.item():.4f}")
    print("Аналитически: df/dx=2x+2yz, df/dy=2y+2xz, df/dz=2z+2xy")

    # 2.2 MSE для y_pred = w*x_data + b
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])
    w = torch.tensor(1.5, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)
    y_pred = w * x_data + b
    loss = mse(y_pred, y_true)
    loss.backward()
    print("\n=== 2.2 Градиенты MSE ===")
    print("y_pred:", y_pred.detach().tolist())
    print(f"MSE={loss.item():.4f}; dMSE/dw={w.grad.item():.4f}; dMSE/db={b.grad.item():.4f}")
    # Аналитические формулы: dL/dw = 2/n sum((pred-true)*x), dL/db=2/n sum(pred-true).
    grad_w_analytic = (2 / x_data.numel()) * ((y_pred.detach() - y_true) * x_data).sum()
    grad_b_analytic = (2 / x_data.numel()) * (y_pred.detach() - y_true).sum()
    print(f"Проверка формул: dL/dw={grad_w_analytic.item():.4f}; dL/db={grad_b_analytic.item():.4f}")

    # 2.3 f(x) = sin(x^2 + 1); производная = cos(x^2+1) * 2x.
    x3 = torch.tensor(0.7, requires_grad=True)
    f3 = torch.sin(x3**2 + 1)
    grad_auto, = torch.autograd.grad(f3, x3)
    grad_analytic = torch.cos(x3**2 + 1) * 2 * x3
    print("\n=== 2.3 Цепное правило ===")
    print(f"f={f3.item():.6f}; autograd={grad_auto.item():.6f}; аналитически={grad_analytic.item():.6f}")
    assert math.isclose(grad_auto.item(), grad_analytic.item(), rel_tol=1e-6, abs_tol=1e-6)


if __name__ == "__main__":
    main()
