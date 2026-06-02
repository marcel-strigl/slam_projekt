import csv
import math

error_x = []
error_y = []
error_theta = []

with open("/home/marcel/SLAM_Projekt/Vergleichstabelle.csv", newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter=';')

    for row in reader:

        wheel_x = float(row["wheel_x"].replace(',', '.'))
        wheel_y = float(row["wheel_y"].replace(',', '.'))

        visual_x = float(row["visual_x"].replace(',', '.'))
        visual_y = float(row["visual_y"].replace(',', '.'))

        imu_theta = float(row["imu_thea"].replace(',', '.'))
        visual_theta = float(row["visual_theta"].replace(',', '.'))

        ex = visual_x - wheel_x
        ey = visual_y - wheel_y

        etheta = visual_theta - imu_theta
        etheta = math.atan2(
            math.sin(etheta),
            math.cos(etheta)
        )

        error_x.append(abs(ex))
        error_y.append(abs(ey))
        error_theta.append(abs(etheta))

print("Max Fehler X:", max(error_x))
print("Max Fehler Y:", max(error_y))
print("Max Fehler Theta:", max(error_theta))
print("Max Fehler Theta [Grad]:", math.degrees(max(error_theta)))