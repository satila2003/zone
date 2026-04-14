from sgp4.api import Satrec, jday
import numpy as np
from datetime import datetime
from .models import Satellite

class OrbitCalculator:
    def __init__(self):
        self.satellites = []

    def load_tle_data(self, tle_text, filter_alt=None, alt_tol=50, filter_inc=None, inc_tol=1.0):
        lines = tle_text.strip().split('\n')
        self.satellites = []
        lines = [L.strip() for L in lines if L.strip()]
        
        mu = 3.986004418e14
        R_earth = 6371.0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("1 "):
                l1 = lines[i]; l2 = lines[i+1]; i += 2; name = f"SAT"
            else:
                name = line; l1 = lines[i+1]; l2 = lines[i+2]; i += 3

            try:
                satrec = Satrec.twoline2rv(l1, l2)
                n = satrec.no_kozai / 60.0
                if n > 0:
                    a = (mu / (n ** 2)) ** (1.0 / 3)
                    alt_km = (a / 1000.0) - R_earth
                else:
                    alt_km = 0
                inclination_deg = np.degrees(satrec.inclo) % 360.0

                keep = True
                if filter_alt is not None and abs(alt_km - filter_alt) > alt_tol: keep = False
                if filter_inc is not None and keep and abs(inclination_deg - filter_inc) > inc_tol: keep = False

                if keep:
                    if name == "SAT": name = f"SAT-{satrec.satnum}"
                    sat = Satellite(satrec.satnum, name, l1, l2)
                    sat._sgp4 = satrec
                    sat.altitude = float(alt_km)
                    sat.inclination = float(inclination_deg)
                    sat.raan = float(np.degrees(satrec.nodeo) % 360.0)
                    # 初始化坐标
                    sat.position = np.array([0.0, 0.0, 0.0])     
                    sat.position_eci = np.array([0.0, 0.0, 0.0]) 
                    self.satellites.append(sat)
            except:
                pass     
        return len(self.satellites)

    def propagate(self, current_time: datetime):
        jd, fr = jday(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second)
        gst = self._gstime(jd + fr)
        c, s = np.cos(gst), np.sin(gst)

        for sat in self.satellites:
            e, r, v = sat._sgp4.sgp4(jd, fr)
            if e == 0:
                # 1. 必须保存 ECI 坐标
                sat.position_eci = np.array(r)
                # 2. 保存 ECEF 坐标
                x, y, z = r
                x_ecef = x * c + y * s
                y_ecef = -x * s + y * c
                z_ecef = z
                sat.position = np.array([x_ecef, y_ecef, z_ecef])
            else:
                sat.position = np.array([0.0, 0.0, 0.0])
                sat.position_eci = np.array([0.0, 0.0, 0.0])

    def _gstime(self, jdut1):
        tut1 = (jdut1 - 2451545.0) / 36525.0
        temp = -6.2e-6 * tut1**3 + 0.093104 * tut1**2 + \
               (876600.0*3600 + 8640184.812866) * tut1 + 67310.54841
        temp = (temp * (np.pi/180.0) / 240.0) % (2*np.pi)
        if temp < 0: temp += 2*np.pi
        return temp