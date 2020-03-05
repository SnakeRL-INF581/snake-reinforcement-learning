"""
Snake Eater
Made with PyGame
"""
import pygame

# RGB colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED   = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

class Visual():
    
    def __init__(self, size_x, size_y, zoom=10):
        self._frame_size_x = size_x * zoom
        self._frame_size_y = size_y * zoom
        self._zoom = zoom
        self._window = pygame.display.set_mode(
            (self._frame_size_x, self._frame_size_y))
        self._window.fill(BLACK)
        self._fps_controller = pygame.time.Clock() # FPS (frames per second)

        # Checks for errors encountered
        check_errors = pygame.init()
        
        if check_errors[1] > 0:
            e_msg = '[!] %d errors when initializing game, exiting...'.format(
                    check_errors[1])
            raise Exception(e_msg)
        else:
            print('[+] Game successfully initialized')

        # Initialize game window
        pygame.display.set_caption('Snake Eater')

    '''
    @param snake_body
        array of snake body block positions (position is an array of length 2)
    @param food
        position of the food
    @param score
        updated score
    '''
    def draw(self, snake_body, food_pos, score, speed):
        self._window.fill(BLACK)
        # Snake head
        snake_head = snake_body[0]
        pygame.draw.rect(self._window, RED, 
            pygame.Rect(snake_head[0]*self._zoom, snake_head[1]*self._zoom, 
            self._zoom, self._zoom))
        
        # Snake body
        for pos in snake_body[1:]:
            pygame.draw.rect(self._window, GREEN, 
                pygame.Rect(pos[0]*self._zoom, pos[1]*self._zoom, 
                self._zoom, self._zoom))
        
        # Snake food
        pygame.draw.rect(self._window, WHITE, 
            pygame.Rect(food_pos[0]*self._zoom, food_pos[1]*self._zoom, 
            self._zoom, self._zoom))
        
        # Score
        score_font = pygame.font.SysFont('consolas', 2*self._zoom)
        score_surface = score_font.render('Score: ' + str(score), True, WHITE)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self._frame_size_x//self._zoom, 
            self._frame_size_y//self._zoom//2)
        
        self._window.blit(score_surface, score_rect)
        # pygame.display.flip()

        # Refresh game screen
        pygame.display.update()

        self._fps_controller.tick(speed)
