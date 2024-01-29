from model.maximin_distance import MaximinDistance
from output_handler.drawer import draw_single_plot


def main():
    for number in range(15, 17):
        model = MaximinDistance(number)
        coords, dist = model.solve()
        draw_single_plot(f'data/output/{number}', coords, dist)
        print(coords)
        print(dist)
    return


if __name__ == "__main__":
    main()
