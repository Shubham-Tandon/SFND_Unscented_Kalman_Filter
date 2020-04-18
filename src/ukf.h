#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const VectorXd &z);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const VectorXd &z, double delta_t);

  /**
   * Generate Augmented Points using the current state and process covariance
   * matrix. This function is used for both prediction step as well as RADAR
   * measurement update step.
   * @param {double} delta_t the change in time (in seconds) between the last
   * measurement and this one.
   **/
  void GenerateAugPoints(double delta_t);

  /*
  * Initialize the state vector with an initial value using Radar data.
  * @param {MeasurementPackage} meas_package
  */
  void InitializeRadar(MeasurementPackage measurement_pack);


  /*
  * Initialize the state vector with an initial value using Lidar data.
  * @param {MeasurementPackage} meas_package
  */
  void InitializeLidar(MeasurementPackage measurement_pack);


  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  ///*Process covariance matrix
  MatrixXd Q_;

  ///*Measurement covariance matrix for radar
  MatrixXd R_radar_;

  ///*Measurement covariance matrix for laser
  MatrixXd R_laser_;

  ///* measurement matrix
  MatrixXd H_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

    ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  // previous timestamp
  long previous_timestamp_;
};

#endif  // UKF_H