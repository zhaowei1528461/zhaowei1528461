import math
import traci


class safe:
    def __init__(self, ego_ID, amax, stop_line):
        self.ego_ID = ego_ID
        self.amax = amax
        self.stop_line = stop_line

    def S13(self, ego_speed, ego_x, ego_y, other_v,other_x, other_y, length, width, amax, stop_line):
        if other_x < ego_x - (length+width)/2 and ego_y < other_y - (length+width)/2 and other_v != 0 and ego_y > other_y - (length + width)/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2 and other_x >= -stop_line-length/2 - 7:
            other_time_min = (ego_x - other_x - (length+width)/2) / other_v
            other_time_max = (ego_x - other_x + (length+width)/2) / other_v
            if ego_speed != 0:
                ego_time_min = (other_y - ego_y - (length+width)/2 - 2) / ego_speed
                ego_time_max = (other_y - ego_y + (length+width)/2) / ego_speed
            elif other_y - (length+width)/2 - ego_y <= 2:
                ego_time_min = -100
                ego_time_max = 100
            else:
                ego_time_min = 100
                ego_time_max = -100
            if ego_time_max > other_time_min - 1.5 and ego_time_min < other_time_max + 1.5:
                # print('1_1')
                return False
            else:
                return True
        elif other_x >= ego_x - (length+width)/2 and other_x <= ego_x + (length+width)/2 and ego_y < other_y - (length+width)/2 and ego_y > other_y - (length + width)/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2:
            mingap = ego_speed*0.2 + ego_speed**2/(2*amax)
            gap = other_y - ego_y - (length+width)/2
            if mingap > gap - 2:
                # print('1_2')
                return False
            else:
                return True
        else:
            return True

    def S31(self, ego_speed, ego_x, ego_y, other_v,other_x, other_y, length, width, amax, stop_line):
        if other_x > ego_x + (length+width)/2 and ego_y < other_y - (length+width)/2 and other_v != 0 and ego_y > other_y - (length + width)/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2 and other_x <= stop_line+length/2 + 10.2:
            other_time_min = (other_x - ego_x - (length+width)/2) / other_v
            other_time_max = (other_x - ego_x + (length+width)/2) / other_v
            if ego_speed != 0:
                ego_time_min = (other_y - ego_y - (length+width)/2 - 2) / ego_speed
                ego_time_max = (other_y - ego_y + (length+width)/2) / ego_speed
            elif other_y - (length+width)/2 - ego_y <= 2:
                ego_time_min = -100
                ego_time_max = 100
            else:
                ego_time_min = 100
                ego_time_max = -100
            if ego_time_max > other_time_min - 1.5 and ego_time_min < other_time_max + 1.5:
                # print('2_1')
                return False
            else:
                return True
        elif other_x >= ego_x - (length+width)/2 and other_x <= ego_x + (length+width)/2 and ego_y < other_y - (length+width)/2 and ego_y > other_y - (length + width)/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2:
            mingap = ego_speed*0.2 + ego_speed**2/(2*amax)
            gap = other_y - ego_y - (length+width)/2
            if mingap > gap - 2:
                # print('2_2')
                return False
            else:
                return True
        else:
            return True

    def S23(self, ego_speed, ego_x, ego_y, other_v,other_x, other_y, length, width, amax, stop_line):
        r0 = stop_line + length/2 + ego_x
        x0 = stop_line + length/2
        y0 = stop_line + length/2
        if other_x == x0:
            other_a = 1.5708
        else:
            other_a = math.atan((y0 - other_y) / (x0 - other_x))
        a0 = length/2/r0
        y1 = -((r0 - width/2) ** 2 - (ego_x - width/2 - x0) ** 2) ** 0.5 + y0
        a1 = math.atan((y0 - y1) / (x0 - (ego_x - width/2))) - a0
        y2 = -((r0 + width / 2) ** 2 - (ego_x + width / 2 - x0) ** 2) ** 0.5 + y0
        a2 = math.atan((y0 - y2) / (x0 - (ego_x + width / 2))) + a0

        if other_a <= a1 and other_y >= 2 and other_v != 0 and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 3 and ego_y < y2 - length/2 and other_y <= stop_line+length/2 + 12:
            if other_y < stop_line+length/2:
                other_time_min = (a1 - other_a) / (other_v / r0)
                other_time_max = (a2 - other_a) / (other_v / r0)
            else:
                other_time_min = (other_y - (stop_line+length/2)) / other_v + a1 / (other_v / r0)
                other_time_max = (other_y - (stop_line+length/2)) / other_v + a2 / (other_v / r0)
            if ego_speed != 0:
                ego_time_min = (y2 - ego_y - length/2 - 3) / ego_speed
                ego_time_max = (y1 - ego_y + length/2) / ego_speed
            elif y2 - length/2 - ego_y <= 3:
                ego_time_min = -100
                ego_time_max = 100
            else:
                ego_time_min = 100
                ego_time_max = -100
            if ego_time_max > other_time_min - 1.5 and ego_time_min < other_time_max + 1.5:
                # print('3_1')
                return False
            else:
                return True
        elif other_a >= a1 and other_a <= a2 + 0.1 and (other_v != 0 or other_a > 0.45) and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 3 and ego_y < y2 - length/2:
            mingap = ego_speed*0.2 + ego_speed**2/(2*amax)
            gap = y2 - ego_y - length/2
            if mingap > gap - 3:
                # print('3_2')
                return False
            else:
                return True
        else:
            return True

    def S12(self, ego_speed, ego_x, ego_y, other_v,other_x, other_y, length, width, amax, stop_line):
        r0 = stop_line + length / 2 + ego_x
        x0 = -(stop_line + length / 2)
        y0 = stop_line + length / 2
        if other_x == -(stop_line+length/2):
            other_a = -1.5708
        else:
            other_a = math.atan((y0 - other_y) / (x0 - other_x))
        a0 = length / 2 / r0
        y1 = y0
        a1 = 0
        y2 = -((r0 + width / 2) ** 2 - ((ego_x - width / 2) - x0) ** 2) ** 0.5 + y0
        a2 = math.atan((y0 - y2) / (x0 - (ego_x - width / 2))) - a0
        if other_a <= a2 and other_v != 0 and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2 and ego_y < y2 - length/2 and other_x >= -(stop_line + length/2) - 7:
            if other_x > -(stop_line + length / 2):
                other_time_min = (a2 - other_a) / (other_v / r0)
                other_time_max = (- other_a) / (other_v / r0)
            else:
                other_time_min = (-(stop_line + length / 2) - other_x) / other_v + (1.571+a2) / (other_v / r0)
                other_time_max = (-(stop_line + length / 2) - other_x) / other_v + 1.571 / (other_v / r0)
            if ego_speed != 0:
                ego_time_min = (y2 - ego_y - length/2 - 2) / ego_speed
                ego_time_max = ((stop_line + length / 2) - ego_y) / ego_speed
            elif y2 - length/2 - ego_y <= 2:
                ego_time_min = -100
                ego_time_max = 100
            else:
                ego_time_min = 100
                ego_time_max = -100
            if ego_time_max > other_time_min - 1.5 and ego_time_min < other_time_max + 1.5:
                # print('4_1')
                return False
            else:
                return True
        elif other_a >= a2 and other_y <= y1 -length/2 and other_x < ego_x and other_x > -(stop_line + length / 2) and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2 and ego_y < y2 - length/2:
            mingap = ego_speed*0.2 + ego_speed**2/(2*amax)
            gap = y2 - ego_y - length/2
            if mingap > gap - 2:
                # print('4_2')
                return False
            else:
                return True
        else:
            return True

    def S34(self, ego_speed, ego_x, ego_y, other_v,other_x, other_y, length, width, amax, stop_line):
        r0 = stop_line + length / 2 + ego_x
        x0 = stop_line + length / 2
        y0 = -(stop_line + length / 2)
        if other_x == (stop_line + length / 2):
            other_a = -1.5708
        else:
            other_a = math.atan((y0 - other_y) / (x0 - other_x))
        a0 = length / 2 / r0
        y1 = ((r0 + width / 2) ** 2 - ((ego_x + width / 2) - x0) ** 2) ** 0.5 + y0
        a1 = math.atan((y0 - y1) / (x0 - (ego_x + width / 2))) - a0
        y2 = ((r0 - width / 2) ** 2 - ((ego_x - width / 2) - x0) ** 2) ** 0.5 + y0
        a2 = math.atan((y0 - y2) / (x0 - (ego_x - width / 2))) + a0
        if (other_a <= a1 or other_x >= (stop_line + length / 2)) and other_v != 0 and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 3 and ego_y < y2 - length/2 and other_x <= (stop_line + length / 2) + 15.2:
            if other_x < (stop_line + length / 2):
                other_time_min = (a1 - other_a) / (other_v / r0)
                other_time_max = (a2 - other_a) / (other_v / r0)
            else:
                other_time_min = (other_x - (stop_line + length / 2)) / other_v + (1.5708+a1) / (other_v / r0)
                other_time_max = (other_x - (stop_line + length / 2)) / other_v + (1.5708+a2) / (other_v / r0)
            if ego_speed != 0:
                ego_time_min = (y2 - ego_y - length/2 - 3) / ego_speed
                ego_time_max = (y1 - ego_y + length/2) / ego_speed
            elif y2 - length/2 - ego_y <= 3:
                ego_time_min = -100
                ego_time_max = 100
            else:
                ego_time_min = 100
                ego_time_max = -100
            if ego_time_max > other_time_min - 1.5 and ego_time_min < other_time_max + 1.5:
                # print('5_1')
                return False
            else:
                return True
        elif other_a >= a1 and other_x < (stop_line + length / 2) and other_x > -1.6 and other_a <= a2 + 0.1 and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 3 and ego_y < y2 - length/2:
            mingap = ego_speed*0.2 + ego_speed**2/(2*amax)
            gap = y2 - ego_y - length/2
            if mingap > gap - 3:
                # print('5_2')
                return False
            else:
                return True
        else:
            return True

    def S32(self, ego_speed, ego_x, ego_y, other_v,other_x, other_y, length, width, amax, stop_line):
        r0 = stop_line + length / 2 - ego_x
        x0 = stop_line + length / 2
        y0 = stop_line + length / 2
        if other_x == x0:
            other_a = 1.5708
        else:
            other_a = math.atan((y0 - other_y) / (x0 - other_x))
        a0 = length / 2 / r0
        y1 = y0
        a1 = 0
        y2 = -((r0 + width / 2) ** 2 - ((ego_x + width / 2) - x0) ** 2) ** 0.5 + y0
        a2 = math.atan((y0 - y2) / (x0 - (ego_x + width / 2))) + a0
        if (other_a >= a2 or other_x >= (stop_line + length/2)) and other_v != 0 and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2 and ego_y < y2 - length/2 and other_x <= (stop_line + length/2) + 10.2:
            if other_x < (stop_line + length / 2):
                other_time_min = (other_a - a2) / (other_v / r0)
                other_time_max = other_a / (other_v / r0)
            else:
                other_time_min = (other_x - (stop_line + length / 2)) / other_v + (1.571-a2) / (other_v / r0)
                other_time_max = (other_x - (stop_line + length / 2)) / other_v + 1.571 / (other_v / r0)
            if ego_speed != 0:
                ego_time_min = (y2 - ego_y - length/2 - 2) / ego_speed
                ego_time_max = ((stop_line + length / 2) - ego_y) / ego_speed
            elif y2 - length/2 - ego_y <= 2:
                ego_time_min = -100
                ego_time_max = 100
            else:
                ego_time_min = 100
                ego_time_max = -100
            if ego_time_max > other_time_min - 1.5 and ego_time_min < other_time_max + 1.5:
                # print('4_1')
                return False
            else:
                return True
        elif other_a <= a2 and other_y <= y1 -length/2 and other_x < (stop_line + length / 2) and ego_y > y2 - length/2 - (1.2 * (ego_speed ** 2) / (2*amax)) - 2 and ego_y < y2 - length/2:
            mingap = ego_speed*0.2 + ego_speed**2/(2*amax)
            gap = y2 - ego_y - length/2
            if mingap > gap - 2:
                # print('4_2')
                return False
            else:
                return True
        else:
            return True

    def check(self):
        check = 0
        ego_ID = self.ego_ID
        amax = self.amax
        stop_line = self.stop_line
        ego_x1 , ego_y1 = traci.vehicle.getPosition(ego_ID)
        ego_speed = traci.vehicle.getSpeed(ego_ID)
        length = traci.vehicle.getLength(ego_ID)
        width = traci.vehicle.getWidth(ego_ID)
        angle = traci.vehicle.getAngle(ego_ID)
        angle1 = math.radians(angle)
        ego_x = ego_x1 - math.sin(angle1) * 2.18
        ego_y = ego_y1 - math.cos(angle1) * 2.18
        surrounding_vehicles = traci.vehicle.getIDList()
        for other_vehicle_id in surrounding_vehicles:
            if other_vehicle_id == 'Auto':
                continue  # 跳过�?�?车辆
            # 获取其他车辆的位�?、方向和长�??
            other_pos_x , other_pos_y = traci.vehicle.getPosition(other_vehicle_id)
            other_angle = traci.vehicle.getAngle(other_vehicle_id)
            other_angle1 = math.radians(other_angle)
            other_x = other_pos_x - math.sin(other_angle1) * 2.18
            other_y = other_pos_y - math.cos(other_angle1) * 2.18
            other_v = traci.vehicle.getSpeed(other_vehicle_id)
            other_route = traci.vehicle.getRouteID(other_vehicle_id)

            if other_route == 'Straight12':
                if not self.S12(ego_speed, ego_x, ego_y, other_v, other_x, other_y, length, width, amax, stop_line):
                    check += 1
            elif other_route == 'Straight13':
                if not self.S13(ego_speed, ego_x, ego_y, other_v, other_x, other_y, length, width, amax, stop_line):
                    check += 1
            elif other_route == 'Straight23':
                if not self.S23(ego_speed, ego_x, ego_y, other_v, other_x, other_y, length, width, amax, stop_line):
                    check += 1
            elif other_route == 'Straight31':
                if not self.S31(ego_speed, ego_x, ego_y, other_v, other_x, other_y, length, width, amax, stop_line):
                    check += 1
            elif other_route == 'Straight32':
                if not self.S32(ego_speed, ego_x, ego_y, other_v, other_x, other_y, length, width, amax, stop_line):
                    check += 1
            elif other_route == 'Straight34':
                if not self.S34(ego_speed, ego_x, ego_y, other_v, other_x, other_y, length, width, amax, stop_line):
                    check += 1
            else:
                check = check

        return check