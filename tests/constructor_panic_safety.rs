use stack_algebra::Matrix;
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

struct DropSpy {
    drops: Rc<Cell<usize>>,
}

impl Drop for DropSpy {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn from_fn_drops_every_initialized_value_when_callback_panics() {
    let drops = Rc::new(Cell::new(0));
    let result = catch_unwind(AssertUnwindSafe({
        let drops = Rc::clone(&drops);
        move || {
            let _ = Matrix::<2, 3, DropSpy>::from_fn(|row, column| {
                if (row, column) == (1, 1) {
                    panic!("intentional constructor failure");
                }
                DropSpy {
                    drops: Rc::clone(&drops),
                }
            });
        }
    }));

    assert!(result.is_err());
    // Column-major construction produced (0,0), (1,0), and (0,1) before
    // the callback panicked at (1,1). All three values must be dropped.
    assert_eq!(drops.get(), 3);
}

#[test]
fn from_fn_transfers_all_initialized_values_to_the_completed_matrix() {
    let drops = Rc::new(Cell::new(0));
    let matrix = Matrix::<2, 3, DropSpy>::from_fn(|_, _| DropSpy {
        drops: Rc::clone(&drops),
    });

    assert_eq!(drops.get(), 0);
    drop(matrix);
    assert_eq!(drops.get(), 6);
}
