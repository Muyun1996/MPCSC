import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import pygame as pg
import torch
import time
import imageio
import os
import csv
from random import randint
import platform
system_name = platform.system()
# Simulation animation not displayed on Linux
# if system_name == 'Linux':
os.environ["SDL_VIDEODRIVER"] = "dummy"

'''
PyNBoids - a Boids simulation - github.com/Nikorasu/PyNBoids
Uses numpy array math instead of math lib, more efficient.
Copyright (c) 2021  Nikolaus Stromberg  nikorasu85@gmail.com
'''
FLLSCRN = True          # True for Fullscreen, or False for Window
BOIDZ = 15             # How many boids to spawn, too many may slow fps
WRAP = False            # False avoids edges, True wraps to other side
FISH = False            # True to turn boids into fish
SPEED = 170             # Movement speed
WIDTH = 800            # Window Width (1200)
HEIGHT = 800            # Window Height (800)
BGCOLOR = (23, 27, 31)     # Background color in RGB
# (33, 37, 41)  (254, 250, 224)
FPS = 60                # 30-90
SHOWFPS = False         # show frame rate

class Boid(pg.sprite.Sprite):
    def __init__(self, boidNum, data, width=200, height=200, isFish=False, cHSV=None):
        super().__init__()
        self.is_intervened = False
        self.intervention_action = 0
        self.data = data
        self.bnum = boidNum
        self.maxW = width
        self.maxH = height
        self.image = pg.Surface((15, 15))
        self.image.set_colorkey(0)
        self.color = pg.Color(0)  # preps color so we can use hsva
        self.color.hsva = (randint(0,360), 90, 90) if cHSV is None else cHSV # randint(5,55) #4goldfish
        if isFish:  # (randint(120,300) + 180) % 360  #4noblues
            pg.draw.polygon(self.image, self.color, ((7,0),(12,5),(3,14),(11,14),(2,5),(7,0)), width=3)
            self.image = pg.transform.scale(self.image, (16, 24))
        else : pg.draw.polygon(self.image, self.color, ((7,0), (13,14), (7,11), (1,14), (7,0)))
        self.bSize = 22 if isFish else 17
        self.orig_image = pg.transform.rotate(self.image.copy(), -90)
        self.dir = pg.Vector2(1, 0)  # sets up forward direction
        self.rect = self.image.get_rect(center=(randint(50, self.maxW - 50), randint(50, self.maxH - 50)))
        self.ang = randint(0, 360)  # random start angle, & position ^
        self.pos = pg.Vector2(self.rect.center)
    def update(self, dt, speed, ejWrap=False):
        maxW, maxH = self.maxW, self.maxH
        turnDir = xvt = yvt = yat = xat = 0
        turnRate = 120 * dt  # about 120 seems ok
        margin = 42
        # Make list of nearby boids, sorted by distance
        otherBoids = np.delete(self.data.array, self.bnum, 0)
        array_dists = (self.pos.x - otherBoids[:,0])**2 + (self.pos.y - otherBoids[:,1])**2
        closeBoidIs = np.argsort(array_dists)[:7]
        neiboids = otherBoids[closeBoidIs]
        neiboids[:,3] = np.sqrt(array_dists[closeBoidIs])
        neiboids = neiboids[neiboids[:,3] < self.bSize*12]
        if neiboids.size > 1:  # if has neighborS, do math and sim rules
            yat = np.sum(np.sin(np.deg2rad(neiboids[:,2])))
            xat = np.sum(np.cos(np.deg2rad(neiboids[:,2])))
            # averages the positions and angles of neighbors
            tAvejAng = np.rad2deg(np.arctan2(yat, xat))
            targetV = (np.mean(neiboids[:,0]), np.mean(neiboids[:,1]))
            # if too close, move away from closest neighbor
            if neiboids[0,3] < self.bSize : targetV = (neiboids[0,0], neiboids[0,1])
            # get angle differences for steering
            tDiff = pg.Vector2(targetV) - self.pos
            tDistance, tAngle = pg.math.Vector2.as_polar(tDiff)
            # if boid is close enough to neighbors, match their average angle
            if tDistance < self.bSize*6 : tAngle = tAvejAng
            # computes the difference to reach target angle, for smooth steering
            angleDiff = (tAngle - self.ang) + 180
            if abs(tAngle - self.ang) > 1.2: turnDir = (angleDiff / 360 - (angleDiff // 360)) * 360 - 180
            # if boid gets too close to target, steer away
            if tDistance < self.bSize and targetV == (neiboids[0,0], neiboids[0,1]) : turnDir = -turnDir
        # Avoid edges of screen by turning toward the edge normal-angle
        if not ejWrap and min(self.pos.x, self.pos.y, maxW - self.pos.x, maxH - self.pos.y) < margin:
            if self.pos.x < margin : tAngle = 0
            elif self.pos.x > maxW - margin : tAngle = 180
            if self.pos.y < margin : tAngle = 90
            elif self.pos.y > maxH - margin : tAngle = 270
            angleDiff = (tAngle - self.ang) + 180  # if in margin, increase turnRate to ensure stays on screen
            turnDir = (angleDiff / 360 - (angleDiff // 360)) * 360 - 180
            edgeDist = min(self.pos.x, self.pos.y, maxW - self.pos.x, maxH - self.pos.y)
            turnRate = turnRate + (1 - edgeDist / margin) * (20 - turnRate) #minRate+(1-dist/margin)*(maxRate-minRate)
        if turnDir != 0:  # steers based on turnDir, handles left or right
            self.ang += turnRate * abs(turnDir) / turnDir
            self.ang %= 360  # ensures that the angle stays within 0-360
        # Adjusts angle of boid image to match heading
        self.image = pg.transform.rotate(self.orig_image, -self.ang)
        self.rect = self.image.get_rect(center=self.rect.center)  # recentering fix
        self.dir = pg.Vector2(1, 0).rotate(self.ang).normalize()
        self.pos += self.dir * dt * (speed + (7 - neiboids.size) * 2)  # movement speed
        # Optional screen wrap
        Surf_rect = pg.Rect(0, 0, self.maxW, self.maxH)
        if ejWrap and not Surf_rect.contains(self.rect):
            if self.rect.bottom < 0 : self.pos.y = maxH
            elif self.rect.top > maxH : self.pos.y = 0
            if self.rect.right < 0 : self.pos.x = maxW
            elif self.rect.left > maxW : self.pos.x = 0
        # Actually update position of boid
        self.rect.center = self.pos
        if self.is_intervened:
            self.ang = (self.ang + self.intervention_action + 360) % 360
            self.is_intervened = False
        # Finally, output pos/ang to array
        self.data.array[self.bnum,:3] = [self.pos[0], self.pos[1], self.ang]

class BoidArray():  # Holds array to store positions and angles
    """
    This class (data) is shared among all the boids
    """
    def __init__(self, boids_num):
        self.array = np.zeros((boids_num, 4), dtype=float)

class InitBoids():
    def __init__(self, boids_num):
        self.init_nBoids_rect = []
        self.init_nBoids_ang = []
        self.init_nBoids_pos = []

    def add(self, rect, ang, pos):
        self.init_nBoids_rect.append(copy.deepcopy(rect))
        self.init_nBoids_ang.append(copy.deepcopy(ang))
        self.init_nBoids_pos.append(copy.deepcopy(pos))

    def get_rect(self, boids_id):
        return copy.deepcopy(self.init_nBoids_rect[boids_id])

    def get_ang(self, boids_id):
        return copy.deepcopy(self.init_nBoids_ang[boids_id])

    def get_pos(self, boids_id):
        return copy.deepcopy(self.init_nBoids_pos[boids_id])

class BoidsDynamics:
    def __init__(self, T=150, boids_num=2, driver_boids_num=1, width=500, height=500, speed=170, dt=0.02, is_wrap=True, seed=1111):
        """
        :param T: simulation time horizen
        :param boids_num:
        :param width:  must larger than 100
        :param height: must larger than 100
        :param speed:
        :param dt:
        :param is_wrap: False avoids edges, True wraps to other side
        :param seed:
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.T = T
        self.boids_num = boids_num
        self.driver_boids_num = driver_boids_num
        self.driver_boids_list = sorted(random.sample(range(0, boids_num), driver_boids_num))
        self.observe_boids_num = boids_num  # fully observable
        self.width = width
        self.height = height
        self.dataArray = BoidArray(self.boids_num)
        self.nBoids = pg.sprite.Group()
        self.dt = dt
        self.speed = speed
        self.is_wrap = is_wrap
        self.init_nBoids = InitBoids(self.boids_num)
        for i in range(self.boids_num):
            # (215,67,34)   (74, 48, 42) (67, 40, 24)
            if i in self.driver_boids_list:
                self.nBoids.add(Boid(i, data=self.dataArray, width=width, height=width, isFish=FISH, cHSV=(360, 100, 82)))
            else:
                self.nBoids.add(Boid(i, data=self.dataArray, width=width, height=width, isFish=FISH, cHSV=(202, 55, 83)))
            self.init_nBoids.add(self.nBoids.sprites()[i].rect,
                                 self.nBoids.sprites()[i].ang,
                                 self.nBoids.sprites()[i].pos)
        self.control_matrix = np.zeros([self.driver_boids_num, self.boids_num])
        for idx, boid in enumerate(self.driver_boids_list):
            self.control_matrix[idx][boid] = 1
        self.display = True
        self.random_init = False
        self.reward_list = []
        self.render_num = 1

    def get_new_state(self, boid):
        sin_x = np.sin((2 * np.pi * boid.pos.x) / boid.maxW)
        cos_x = np.cos((2 * np.pi * boid.pos.x) / boid.maxW)
        sin_y = np.sin((2 * np.pi * boid.pos.y) / boid.maxH)
        cos_y = np.cos((2 * np.pi * boid.pos.y) / boid.maxH)
        ang = boid.ang / 360.0
        return [sin_x, cos_x, sin_y, cos_y, ang]

    def get_original_state_from_obs(self, obs):
        original_obs = obs.reshape(-1, 5)

        state = []
        for i in range(original_obs.shape[0]):
            boid_obs = original_obs[i]

            radian_x = np.arctan2(boid_obs[0], boid_obs[1])
            if radian_x < 0:
                radian_x += 2 * np.pi
            pos_x = radian_x * self.nBoids.sprites()[0].maxW / (2 * np.pi)

            radian_y = np.arctan2(boid_obs[2], boid_obs[3])
            if radian_y < 0:
                radian_y += 2 * np.pi
            pos_y = radian_y * self.nBoids.sprites()[0].maxH / (2 * np.pi)

            ang = boid_obs[4] * 360

            state.append([pos_x, pos_y, ang])

        state = np.array(state)
        return state

    def set_state_from_obs(self, obs):
        state = self.get_original_state_from_obs(obs)
        for id, boid in enumerate(self.nBoids.sprites()):
            boid.rect = boid.image.get_rect(center=(state[id, 0], state[id, 1]))
            boid.pos =  pg.Vector2((state[id, 0], state[id, 1]))
            boid.ang = state[id, 2]
            boid.data.array[boid.bnum,:3] = [state[id, 0], state[id, 1], state[id, 2]]
        self.obs = obs

    def reset(self, random_init=False, display=True):
        self.random_init = random_init
        self.display = display
        if self.display:
            pg.init()
            self.screen = pg.display.set_mode((self.width, self.height), pg.SCALED)
            self.frames = []

        # Initialize the data of the boids population
        init_dataArray = BoidArray(self.boids_num)
        for id, boid in enumerate(self.nBoids.sprites()):
            if self.random_init:
                boid.rect = boid.image.get_rect(center=(randint(50, boid.maxW - 50), randint(50, boid.maxH - 50)))
                boid.ang = randint(0, 360)  # random start angle, & position ^
                boid.pos = pg.Vector2(boid.rect.center)
                boid.data = init_dataArray
            else:
                boid.rect = self.init_nBoids.get_rect(id)
                boid.ang = self.init_nBoids.get_ang(id)
                boid.pos = self.init_nBoids.get_pos(id)
                boid.data = init_dataArray

        # get obs
        boids_list = self.nBoids.sprites()
        state_list = []
        for boid in boids_list:
            boid_state = self.get_new_state(boid)
            state_list.append(boid_state)
        state = np.array(state_list)
        state = torch.from_numpy(state).view(-1)
        self.obs = state.numpy()
        self.time_step = 0
        self.done = False
        self.reward_list = []
        return self.obs

    def reward(self, state_np):
        # Get the orientation of all individuals
        angle_degrees = state_np[:, 4] * 360
        angle_radians = np.deg2rad(angle_degrees)
        # Convert the orientations to vectors on the complex plane
        vectors = np.array([np.cos(angle_radians), np.sin(angle_radians)]).T
        # Calculate the magnitude of the sum of vectors
        vector_sum_length = np.linalg.norm(np.sum(vectors, axis=0))
        # Calculate the Order Parameter
        if self.boids_num > 0:
            order_parameter = vector_sum_length / self.boids_num
        else:
            order_parameter = 0.0
        return order_parameter

    def step(self, action=None):
        """
        :param action: [driver_boids_num, 1]
        :return:
        """
        if action is not None:
            if not torch.is_tensor(action):
                action = torch.from_numpy(action)
            action = action.view(-1)
            action = 45.0 * action @ self.control_matrix
            for boid in self.nBoids.sprites():
                if boid.bnum in self.driver_boids_list:
                    boid.intervention_action = action[boid.bnum]
                    boid.is_intervened = True if self.render_num != 0 else False
                    # boid.ang = (boid.ang + action[boid.bnum] + 360) % 360
                    # boid.data.array[boid.bnum, 2] = boid.ang
        self.nBoids.update(self.dt, self.speed, self.is_wrap)
        if self.display:
            self.frames.append(np.array(pg.surfarray.array3d(self.screen)))
        boids_list = self.nBoids.sprites()
        state_list = []
        for boid in boids_list:
            boid_state = self.get_new_state(boid)
            state_list.append(boid_state)
        state_np = np.array(state_list)
        state = torch.from_numpy(state_np).view(-1)
        self.obs = state.numpy()
        reward = self.reward(state_np)
        self.reward_list.append(reward)
        self.time_step += 1
        if self.time_step >= self.T:
            self.done = True
        info = {}
        return self.obs, reward, self.done, info

    def render(self, log_fig_dir=None, is_show=False):
        log_fig_dir = os.getcwd() + "/log_ppo"
        if (log_fig_dir is not None) and (not os.path.exists(log_fig_dir)):
            os.makedirs(log_fig_dir)
        if self.display:
            if self.done and log_fig_dir:
                imageio.mimsave(log_fig_dir + '/boids_animation_{}'.format(self.render_num) + '.gif', self.frames, duration=30)
            self.screen.fill(BGCOLOR)
            self.nBoids.draw(self.screen)
            pg.display.update()
        if self.time_step == self.T:
            total_reward_tensor = "{:.1f}".format(sum(self.reward_list))
            plt.plot(self.reward_list)
            plt.savefig(log_fig_dir + '/boids_reward_{}_rt{}.jpg'.format(self.render_num, total_reward_tensor)) if log_fig_dir else None
            plt.show() if is_show else None
            plt.close()
            self.render_num += 1

            # log_fig_dir = "results"
            # save order parameter data
            if log_fig_dir:
                csv_file = log_fig_dir + "/boids_order_parameter_data_{}_rt{}.csv".format(self.render_num, total_reward_tensor)
                reward_list = [[np_item.item()] for np_item in self.reward_list]
                with open(csv_file, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(reward_list)


def test():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT), pg.SCALED)
    dataArray = BoidArray(BOIDZ)
    nBoids = pg.sprite.Group()
    for n in range(BOIDZ):
        nBoids.add(Boid(n, dataArray, width=WIDTH, height=HEIGHT, isFish=FISH))

    for i in range(10000):
        dt = 0.02
        screen.fill(BGCOLOR)
        nBoids.update(dt, SPEED, WRAP)
        nBoids.draw(screen)
        pg.display.update()

if __name__ == '__main__':
    #test()
    boids_dynamics = BoidsDynamics(T=150, boids_num=10, driver_boids_num=3, width=500, height=500, speed=170, dt=0.02, is_wrap=True, seed=1111)
    for t in range(2 * boids_dynamics.T):
        if t % boids_dynamics.T == 0:
            boids_dynamics.reset(display=True)
        #action = np.random.random(boids_dynamics.driver_boids_num)
        action = 2 * torch.rand(boids_dynamics.driver_boids_num) - 1
        #action = np.zeros(boids_dynamics.driver_boids_num)
        obs, reward, done, info = boids_dynamics.step(action)
        boids_dynamics.render(log_fig_dir="log",is_show=True)




