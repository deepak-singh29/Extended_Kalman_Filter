#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
            || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){

        VectorXd residual = estimations[i] - ground_truth[i];
//        cout<< "residual";
//        cout<< residual <<endl;
        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj = MatrixXd::Zero(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float val11,val12,val21,val22,val31,val32,val33,val34;
    float row1d,row2d,row3d;

    row2d = px*px + py*py;
    row1d = sqrt(row2d);
    row3d = pow(row2d ,1.5);

    if(row2d < 0.00001)
    {
        cout << "Cannot divide by zero";
    }
    else
    {
        val11 = px / row1d;
        val12 = py / row1d;
        val21 = -py / row2d;
        val22 = px / row2d;
        val31 = py *(vx*py -vy*px) / row3d;
        val32 = px *(vy*px - vx*py) / row3d;
        val33 = val11;
        val34 = val12;
        Hj << val11, val12, 0, 0,
                val21, val22, 0, 0,
                (py *(vx*py -vy*px) / row3d), val32, val33, val34;
    }
	//check division by zero

	//compute the Jacobian matrix

	return Hj;
}
