
import matplotlib.pyplot as plt
import numpy as np
import pygame

from Agent import Agent
from limo import Limo
from visualization import Visualization, Vehicle, Object

plt.style.use("ggplot")

def make_Ravnkloa(scale=1, height=1080, width=1920, scale_boat_phys=1, pixels_per_unit=6, viz=True, dt=1.0):
    coordinates = np.array([
            [127.893567, 150.842699],
            [116.813103, 130.938903],
            [86.6496169, 150.021924],
            [1.69939146, 149.406343],
            [1.69939146, 138.120685], 
            [88.2911671, 70.2015434],
            [86.6496169, 50.0925529],
            [95.8833370, 37.1653446],
            [112.093646, 23.4173613],
            [107.784576, 14.3888349],
            [119.275428, 6.79666503],
            [134.254574, 15.2096101],
            [162.981703, 7.20705259],
            [171.599842, 0.230464029],
            [277.479833, 1.05123915],
            [278.505802, 31.0095312],
            [262.911075, 34.9082131],
            [201.763328, 65.4820865],
            [205.867204, 76.5625506],
            [191.708833, 83.9495268],
            [182.680306, 73.4846439],
            [173.036199, 77.9989071],
            [174.472555, 87.4378210],
            [167.495967, 93.1832469],
            [168.932323, 98.9286728],
            [143.693488, 114.728594],
            [150.259689, 144.071305],
            [127.893567, 150.842699]
    ])

    car = Vehicle(
        np.array([90, 90]),
        length=5.0,
        width=3.0,
        heading=0,
        tau_steering=0.4,
        tau_throttle=0.4,
        dt=dt,
    )
    car1 = Vehicle(
        np.array([121, 68]),
        length=5.0,
        width=2.5,
        heading=3*np.pi/2,
        tau_steering=0.2, 
        tau_throttle=0.2,
        dt=dt,
    )
    car2 = Vehicle(
        np.array([82, 104]), 
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car3 = Vehicle(
        np.array([107, 75]),
        length=5.0, 
        width=2.5,
        heading=3*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car4 = Vehicle(
        np.array([86, 126]),
        length=5.0,
        width=2.5, 
        heading=7*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car5 = Vehicle(
        np.array([111, 54]),
        length=5.0,
        width=2.5,
        heading=3*np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car6 = Vehicle(
        np.array([161, 40]),
        length=5.0,
        width=2.5,
        heading=np.pi/2, 
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car7 = Vehicle(
        np.array([136, 98]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car8 = Vehicle(
        np.array([189, 65]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car9 = Vehicle(
        np.array([142, 32]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car10 = Vehicle(
        np.array([25, 128]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car11 = Vehicle(
        np.array([175, 65]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car12 = Vehicle(
        np.array([68, 112]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car13 = Vehicle(
        np.array([155, 45]),
        length=5.0,
        width=2.5,
        heading=3*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car14 = Vehicle(
        np.array([100, 45]),
        length=5.0,
        width=2.5,
        heading=5*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car15 = Vehicle(
        np.array([170, 60]),
        length=5.0,
        width=2.5,
        heading=7*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car16 = Vehicle(
        np.array([145, 75]),
        length=5.0,
        width=2.5,
        heading=np.pi,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car17 = Vehicle(
        np.array([140, 20]),
        length=5.0,
        width=2.5,
        heading=0,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car18 = Vehicle(
        np.array([220, 30]),
        length=5.0,
        width=2.5,
        heading=np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car19 = Vehicle(
        np.array([200, 40]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car20 = Vehicle(
        np.array([180, 40]),
        length=5.0,
        width=2.5,
        heading=3*np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car21 = Vehicle(
        np.array([140, 54]),
        length=5.0,
        width=2.5,
        heading=3*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car22 = Vehicle(
        np.array([110, 84]),
        length=5.0,
        width=2.5,
        heading=5*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car23 = Vehicle(
        np.array([195, 45]),
        length=5.0,
        width=2.5,
        heading=7*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car24 = Vehicle(
        np.array([105, 110]),
        length=5.0,
        width=2.5,
        heading=np.pi,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car25 = Vehicle(
        np.array([150, 25]),
        length=5.0,
        width=2.5,
        heading=0,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car26 = Vehicle(
        np.array([105, 35]),
        length=5.0,
        width=2.5,
        heading=np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car27 = Vehicle(
        np.array([100, 80]),
        length=5.0,
        width=2.5,
        heading=np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car28 = Vehicle(
        np.array([125, 95]),
        length=5.0,
        width=2.5,
        heading=3*np.pi/2,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car29 = Vehicle(
        np.array([150, 55]),
        length=5.0,
        width=2.5,
        heading=3*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )
    car30 = Vehicle(
        np.array([95, 70]),
        length=5.0,
        width=2.5,
        heading=5*np.pi/4,
        tau_steering=0.2,
        tau_throttle=0.2,
        dt=dt,
    )


    # Define the outer rim (harbor boundary)
    outer_rim = Object(
        center=np.array([0, 0]),
        vertices=coordinates,
    )

    objects = [
        car,
        car1,
        car2,
        car3,
        car4,
        car5,
        car6,
        car7,
        car8,
        car9,
        car10,
        car11,
        car12,
        car13,
        car14,
        car15,
        car16,
        car17,
        car18,
        car19,
        car20,
        car21,
        car22,
        car23,
        car24,
        car25,
        car26,
        car27,
        car28,
        car29,
        car30,
        outer_rim
    ]
    cars = [car, car1, car2, car3, car4, car5, car6, car7, car8, car9, car10, 
            car11, car12, car13, car14, car15, car16, car17, car18, car19, car20,
            car21, car22, car23, car24, car25, car26, car27, car28, car29, car30
            ]

    if viz:
        MAP_DIMENSIONS = (height * scale, width * scale)
        gfx = Visualization(
            MAP_DIMENSIONS,
            pixels_per_unit=pixels_per_unit,
            map_img_path="graphics/test_map_2.png",
        )  # Also initializes the display
        return gfx, objects, cars
    else:
        return objects, cars

def driving_with_many_boats():
    # Create a visualizer
    divider = 10
    dt = 1 / divider
    gfx, objects, cars = make_Ravnkloa(
        scale=1, scale_boat_phys=5, height=1080, width=1920, dt=dt
    )
    # gfx, objects, cars = map_tube_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    # gfx, objects, cars = map_lanes_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)
    # gfx, objects, cars = map_city_block_multi(scale=1, height=1080, width=1920, pixels_per_unit=10)

    # Spawn drivers
    alpha_max = 0.8  # boat
    v_max = 3.0  # 6
    v_min = -1.5
    limos = []

    # Spawn milliAmpere
    agent = Agent(0.5, -0.2, alpha_max)
    limo = Limo(vehicle=cars[0], driver=agent)
    limos.append(limo)
    for car in cars[1:]:
        agent = Agent(v_max, v_min, alpha_max)
        # Make it a limo!
        limo = Limo(vehicle=car, driver=agent)
        limos.append(limo)

    # Sets certain parameters
    steps = 8000
    # TODO: move these params to the agent?
    alpha_refs = np.zeros(len(cars))
    v_refs = np.ones(len(cars)) * 2
    for i in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Press x button
                exit()

        ##############
        # Visualize! #
        ##############
        gfx.clear_canvas()
        gfx.draw_all_objects(objects)

        # Generate circogram
        N = 36
        horizon = 500
        # Circograms!
        DCs = []
        SCs = []
        if i % (divider // 6) == 0:  # Only every 0.1 seconds
            for n, car in enumerate(cars):
                static_circogram = car.static_circogram_2(
                    N, objects[0:n] + objects[n + 1 :], horizon
                )
                dynamic_circogram = car.dynamic_cicogram_2(
                    static_circogram, alpha_refs[n], v_refs[n], seconds=3
                )
                d1, d2, _, _, _ = static_circogram
                car.collision_check(d1, d2)
                # gfx.draw_static_circogram_data(static_circogram, car)
                # gfx.draw_dynamic_circogram_data(dynamic_circogram, static_circogram, risk_threshold=1.5, verbose=False)
                DCs.append(dynamic_circogram)
                SCs.append(static_circogram)

            for n, limo in enumerate(limos):
                if not limo.vehicle.collided:
                    v_refs[n], alpha_refs[n] = limo.driver.determined_driver(
                        DCs[n],
                        SCs[n],
                        v_refs[n],
                        alpha_refs[n],
                        risk_threshold=0.4,
                        stop_threshold=2,
                        dist_wait=10,
                        verbose=False,
                    )
                else:
                    v_refs[n], alpha_refs[n] = (0, 0)

        ##############
        # Kinematics #
        ##############
        for n, limo in enumerate(limos):
            # Run one step
            limo.vehicle.one_step_algorithm(alpha_ref=alpha_refs[n], v_ref=v_refs[n])

        gfx.draw_headings(cars, scale=True)
        gfx.draw_centers(cars)
        gfx.update_display()

if __name__ == "__main__":

    driving_with_many_boats()
