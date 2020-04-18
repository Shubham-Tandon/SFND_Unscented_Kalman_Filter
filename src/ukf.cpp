#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.5;

  is_initialized_ = false;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // previous timestamp
  previous_timestamp_ = 0;

  ///*Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  ///* process covariance matrix
  Q_ = MatrixXd(2, 2);
  Q_ << std_a_*std_a_, 0,
        0            , std_yawdd_*std_yawdd_;

  ///* measurement matrix
  H_ = MatrixXd(2, n_x_);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

  ///*Measurement covariance matrix for radar
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_, 0                      , 0,
              0                  , std_radphi_*std_radphi_, 0,
              0                  , 0                      , std_radrd_*std_radrd_;

  ///*Measurement covariance matrix for laser
  R_laser_ = MatrixXd(2,2);
  R_laser_ << std_laspx_*std_laspx_ , 0,
              0                     , std_laspy_*std_laspy_ ;

  ///* Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /**
  * Initialization
  */
  if (!is_initialized_) {

    // first measurement
    std::cout << "UKF: " << endl;
    x_ = VectorXd(n_x_);
    x_.fill(0);

    cout << "Kalman Filter Initialization .." << endl;
    previous_timestamp_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      InitializeRadar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      InitializeLidar(meas_package);
    }

    cout << "Kalman Filter Initialized." << endl;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  
  while (delta_t > 0.1)
  {
    Prediction(0.05);
    delta_t -= 0.05;
  }

  Prediction(delta_t);
  previous_timestamp_ = meas_package.timestamp_;


  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package.raw_measurements_, delta_t );
  }

  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    UpdateLidar(meas_package.raw_measurements_);
  }


}

void UKF::Prediction(double delta_t) 
{

  GenerateAugPoints(delta_t);

  ///* Predicting state vector and covariance matrix from the predicted
  ///* sigma points.
  weights_.fill(1.0/(2.0*(lambda_ + n_aug_)));
  weights_(0) = lambda_/(lambda_ + n_aug_);
  x_.fill(0.);
  P_.fill(0.);

  for (size_t i = 0; i < 2 * n_aug_ + 1; i++){
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }

  for (size_t i = 0; i < 2 * n_aug_ + 1; i++){

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }


}

void UKF::UpdateLidar(const VectorXd &z) 
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  ///*LIDAR measurement dimension
  int n_z_ = 2;

  MatrixXd R_     = R_laser_;;
  VectorXd z_pred = H_ * x_;
  VectorXd y      = z - z_pred;
  MatrixXd Ht     = H_.transpose();
  MatrixXd S      = H_ * P_ * Ht + R_;
  MatrixXd Si     = S.inverse();
  MatrixXd PHt    = P_ * Ht;
  MatrixXd K      = PHt * Si;
  
  //new estimate
  x_          = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I  = MatrixXd::Identity(x_size, x_size);
  P_          = (I - K * H_) * P_;

  NIS_laser_ = y.transpose() * Si * y;

}

void UKF::UpdateRadar(const VectorXd &z, double delta_t)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  //*Radar measurement dimension
  int n_z_ = 3;

  // Create measurement covarince matrix
  MatrixXd R_   = R_radar_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z_);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z_,n_z_);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_);

  for (size_t i = 0; i< 2 * n_aug_ + 1; i++){

    double px      = Xsig_pred_(0,i);
    double py      = Xsig_pred_(1,i);
    double v       = Xsig_pred_(2,i);
    double psi     = Xsig_pred_(3,i);
    double vx      = v*cos(psi);
    double vy      = v*sin(psi);

    // Check division by zero. No radar update if division by zero encountered.
    if ((px*px + py*py) == 0){
      return;
    }

    Zsig.col(i) << sqrt(px*px + py*py),
                    atan2(py, px),
                    (px*vx + py*vy)/sqrt(px*px + py*py);

    z_pred = z_pred + weights_(i)*Zsig.col(i);

  }

  for (size_t i = 0; i < 2 * n_aug_ + 1; i++){

    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_;

  for (size_t i = 0; i< 2 * n_aug_ + 1; i++){

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman Gain
  MatrixXd K = Tc*(S.inverse());
  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K* z_diff;
  P_ = P_ - K*S*(K.transpose());

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}


/**
* Initialize the state vector with an initial value using Radar data.
* @param {MeasurementPackage} meas_package
**/
void UKF::InitializeRadar(MeasurementPackage measurement_pack){
  /**
  Convert radar from polar to cartesian coordinates and initialize state.
  */

  float rho     = measurement_pack.raw_measurements_[0];
  float phi     = measurement_pack.raw_measurements_[1];
  float rho_dot = measurement_pack.raw_measurements_[2];

  float px      = rho*cos(phi);
  float py      = rho*sin(phi);
  float vx      = rho_dot*cos(phi);
  float vy      = rho_dot*sin(phi);
  float v       = sqrt(vx*vx + vy*vy);
  float psi     = 0;
  float psi_dot = 0;

  if (fabs(px) < 0.001 && fabs(py) < 0.001){
    px      = 0.1;
    py      = 0.1;
  }

  x_ << px, py, v, psi, psi_dot;

}


/*
* Initialize the state vector with an initial value using Lidar data.
* @param {MeasurementPackage} meas_package
*/
void UKF::InitializeLidar(MeasurementPackage measurement_pack){

  //set the state with the initial location and zero velocity
  float px      = measurement_pack.raw_measurements_[0];
  float py      = measurement_pack.raw_measurements_[1];
  float vx      = 0;
  float vy      = 0;
  float v       = 0;
  float psi     = 0;
  float psi_dot = 0;

  if (fabs(px) < 0.001 && fabs(py) < 0.001){
    px      = 0.1;
    py      = 0.1;
  }

  x_ << px, py, v, psi, psi_dot;

}


/**
 * Generate Augmented Points using the current state and process covariance
 * matrix. This function is used for both prediction step as well as RADAR
 * measurement update step.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 **/
void UKF::GenerateAugPoints(double delta_t){

  MatrixXd Xsig_aug  = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  VectorXd x_aug     = VectorXd::Zero(n_aug_);
  MatrixXd P_aug     = MatrixXd::Zero(n_aug_, n_aug_);
  VectorXd x_mean    = VectorXd(n_x_);
  VectorXd x_up      = VectorXd(n_x_);
  VectorXd nu_up     = VectorXd(n_x_);

  x_aug.head(n_x_)              = x_;
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug.block(n_x_, n_x_, 2, 2) = Q_;
  MatrixXd A                    = P_aug.llt().matrixL();
  float sq                      = sqrt(lambda_ + n_aug_);
  Xsig_aug.col(0)               = x_aug;

  ///* Generating augmented sigma points *///
  for (size_t i = 0; i<n_aug_; i++){
    Xsig_aug.col(i+1) = x_aug + sq*A.col(i);
    Xsig_aug.col(i+8) = x_aug - sq*A.col(i);
  }

  ///* Calculating predicted sigma points *///
  for (size_t i = 0; i< 2 * n_aug_ + 1; i++){

    double px      = Xsig_aug(0,i);
    double py      = Xsig_aug(1,i);
    double v       = Xsig_aug(2,i);
    double psi     = Xsig_aug(3,i);
    double psi_dot = Xsig_aug(4,i);
    double nu_a    = Xsig_aug(5,i);
    double nu_psid = Xsig_aug(6,i);

    nu_up << 0.5*delta_t*delta_t*cos(psi)*nu_a,
             0.5*delta_t*delta_t*sin(psi)*nu_a,
             delta_t*nu_a,
             0.5*delta_t*delta_t*nu_psid,
             delta_t*nu_psid;

    x_mean = Xsig_aug.block(0,i,n_x_,1);

    if (fabs(psi_dot)>0.001){

      x_up << (v/psi_dot)*( sin(psi + psi_dot*delta_t) - sin(psi)),
              (v/psi_dot)*(-cos(psi + psi_dot*delta_t) + cos(psi)),
              0,
              psi_dot*delta_t,
              0;

      Xsig_pred_.col(i) = x_mean + x_up + nu_up;

    }

    else{
      x_up << v*cos(psi)*delta_t,
              v*sin(psi)*delta_t,
              0,
              psi_dot*delta_t,
              0;

      Xsig_pred_.col(i) = x_mean + x_up + nu_up;
    }
  }
}