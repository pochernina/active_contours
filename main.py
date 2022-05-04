import argparse
from PIL import Image
import numpy as np
from skimage.filters import sobel, gaussian
from scipy.interpolate import RectBivariateSpline, interp1d
import utils

def args_parser():
    parser = argparse.ArgumentParser(description = 'Active Contours')

    parser.add_argument('input_image', type=str)
    parser.add_argument('initial_snake', type=str)  # начальное приближение для контура
    parser.add_argument('output_image', type=str)
    parser.add_argument('alpha', type=float)     # параметр внутр энергии (растяжимость / упругость контура)
    parser.add_argument('beta', type=float)      # параметр внутр энергии (жесткость контура)
    parser.add_argument('tau', type=float)       # шаг градиентного спуска
    parser.add_argument('w_line', type=float)    # параметр внешней энергии (интенсивность)
    parser.add_argument('w_edge', type=float)    # параметр внешней энергии (границы)
    parser.add_argument('kappa', type=float)     # balloon force
    
    args = parser.parse_args()
    return args

def IoU(result, gt):
    im_result = np.array(Image.open(result))
    im_gt = np.array(Image.open(gt))
    im_result //= 255
    im_gt //= 255

    intersection = im_result * im_gt
    union = im_result + im_gt - intersection

    return np.sum(intersection) / np.sum(union)

def active_contours(im, init_snake, alpha, beta, tau, w_line, w_edge, kappa):

    def E_ext(im):
        P_line = gaussian(im, 3)
        P_edge = sobel(P_line)
        E_ext = w_line * P_line + w_edge * P_edge
        return E_ext
    
    def Euler_matrix(n):
        a = np.roll(np.eye(n), -1, axis=1) - 2 * np.eye(n) + np.roll(np.eye(n), -1, axis=0)
        b = np.roll(np.eye(n), -2, axis=1) - 4 * np.roll(np.eye(n), -1, axis=1) + 6 * np.eye(n) - \
            4 * np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -2, axis=0)
        A = -alpha * a + beta * b
        matrix = np.linalg.inv(np.eye(n) + tau * A)
        return matrix

    def reparametrization(snake):
        n = snake.shape[0]
        snake_len = np.zeros(n)
        total_len = 0
        
        for i in range(n):
            total_len += np.linalg.norm(snake[i] - snake[i-1])
            snake_len[i] = total_len

        x = interp1d(snake_len, snake[:, 0])
        y = interp1d(snake_len, snake[:, 1])

        for i in range(snake.shape[0] - 1):
            snake[i][0] = x(i * total_len / snake.shape[0])
            snake[i][1] = y(i * total_len / snake.shape[0])

        return snake


    ext = E_ext(im)

    interpolated_image = RectBivariateSpline(np.arange(ext.shape[1]), np.arange(ext.shape[0]), \
                                             ext.T, kx=2, ky=2, s=0)

    x, y = init_snake[:, 0], init_snake[:, 1]
    n = init_snake.shape[0]
    x_normal = np.zeros(n)
    y_normal = np.zeros(n)
    m = Euler_matrix(n)

    min_dist = 1
    prev_snakes = []

    while min_dist > 0.1:
        fx = interpolated_image(x, y, dx=1, grid=False)
        fy = interpolated_image(x, y, dy=1, grid=False)
        fx /= np.linalg.norm(fx)
        fy /= np.linalg.norm(fy)

        for i in range(n-1):
            x_normal[i] = y[i+1] - y[i]
            y_normal[i] = x[i] - x[i+1]

        xn = m @ (x + tau * (fx + kappa * x_normal))
        yn = m @ (y + tau * (fy + kappa * y_normal))

        x, y = xn, yn
        x[-1], y[-1] = x[0], y[0]

        snake = reparametrization(np.stack([y, x], axis=1))
        snake[snake < 0] = 0
        snake[snake[:, 0] > ext.shape[1] - 1, 0] = ext.shape[1] - 1
        snake[snake[:, 1] > ext.shape[0] - 1, 1] = ext.shape[0] - 1
        x = snake[:, 1]
        y = snake[:, 0]

        if len(prev_snakes) >= 3:
           prev_snakes = prev_snakes[1:]

        for pair in prev_snakes:
            cur_dist = np.average(np.abs(pair[0] - x) + np.abs(pair[1] - y))
            if cur_dist < min_dist:
                min_dist = cur_dist

        prev_snakes.append([x, y])

    return np.stack([x, y], axis=1)


args = args_parser()

im = np.array(Image.open(args.input_image), dtype="float64")
init_snake = np.loadtxt(args.initial_snake)

alpha, beta = args.alpha, args.beta
tau = args.tau
w_line, w_edge = args.w_line, args.w_edge
kappa = args.kappa

snake = active_contours(im, init_snake, alpha, beta, tau, w_line, w_edge, kappa)

# utils.display_snake(im, init_snake, snake)
utils.save_mask(args.output_image, snake, im)

# print('IoU: ', IoU(args.output_image, 'test_data/coins_mask.png'))