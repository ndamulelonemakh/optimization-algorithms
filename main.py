import os
import logging

import conjugatedescent
import coordinatedescent
import gradientdescent

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def _choose_operation(requested_operation: str):
    if requested_operation == 'CONJUGATE_GRADIENT':
        optimizer = conjugatedescent.ConjugateGradientDescent()
    elif requested_operation == 'COORDINATE_DESCENT':
        optimizer = coordinatedescent.CoordinateDescent()
    else:
        optimizer = gradientdescent.GradientDescent()

    log.debug(f'Done activating {type(optimizer).__name__} optimizer for algorithm={requested_operation}')
    return optimizer


def main():
    optimizer = _choose_operation(os.getenv('ALGORITHM'))
    minimiser, min_value, gradient = optimizer.execute()
    result = f'''
    =====Function F(X1, X2) has a local minimum at {minimiser}=========
      - Min Value = {min_value}
      - Slope     = {gradient} 
    '''
    print(result.strip())


if __name__ == '__main__':
    main()
