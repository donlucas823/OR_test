#include <iostream>
#include <vector>
#include "gurobi_c++.h"

using namespace std;

//������׶�³���Ż�����
//min c^T * y + max min b^T * x;

const vector<double> c = { 1, 2 }; // Ŀ�꺯��ϵ��
const vector<vector<double>> A = { {1, 2}, {3, 4} }; // Լ������
const vector<double> d = { 5, 11 }; // Լ���Ҷ���
const vector<double> b = { 1, 1 }; // ������Ŀ�꺯��ϵ��
const vector<vector<double>> G = { {1, 2}, {2, 1} }; // ������Լ������
const vector<double> h = { 10, 10 }; // ������Լ���Ҷ���
const vector<vector<double>> E = { {1, 1}, {1, 1} }; // ��ȷ���Ծ���
const vector<vector<double>> M = { {1, 0}, {0, 1} }; // ��ȷ���Ծ���
const vector<double> u_lower = { 0, 0 }; // ��ȷ���Լ��ϵ��½�
const vector<double> u_upper = { 1, 1 }; // ��ȷ���Լ��ϵ��Ͻ�


double solveMasterProblem(GRBEnv& env, vector<double>& y, double& eta, const vector<vector<double>>& cuts) {
	try {
		GRBModel model(env);//����Gurobiģ��

		// ��������
		int numVars = c.size();
		GRBVar* yVars = model.addVars(numVars, GRB_CONTINUOUS);
		GRBVar etaVar = model.addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS);

		// ����Ŀ�꺯����min c^T y + ��
		GRBLinExpr obj = etaVar;
		for (int i = 0; i < numVars; i++){
			obj += c[i] * yVars[i];
		}
		model.setObjective(obj, GRB_MINIMIZE);

		//���Լ����Ay >= d
		for (size_t i = 0; i < A.size(); i++){
			GRBLinExpr constr = 0;
			for (size_t j = 0; j < numVars; j++){
				constr += A[i][j] * yVars[j];
			}
			model.addConstr(constr >= d[i]);
		}

		//��Ӹ�ƽ��Լ��
		for (const auto &cut :cuts)	{
			GRBLinExpr constr = etaVar;
			for (int i = 0; i < numVars;  i++) {
				constr -= cut[i] * yVars[i];
			}
			model.addConstr(constr >= cut.back());
		}

		//�Ż�ģ��
		model.optimize();

		for (size_t i = 0; i < numVars; i++) {
			y[i] = yVars[i].get(GRB_DoubleAttr_X);
		}
		eta = etaVar.get(GRB_DoubleAttr_X);

		return model.get(GRB_DoubleAttr_ObjVal);
	}

	catch (GRBException& e) {
		cerr << "Error Code = " << e.getErrorCode() << endl;
		cerr << e.getMessage() << endl;
		return GRB_INFINITY;
	}
}

double solveSubProblem(GRBEnv& env, const vector<double>& y, vector<double>& pi, vector<double>& u) {
	try {
		GRBModel model = GRBModel(env);

		int numVars = G[0].size();
		int numConstrs = G.size();
		int numuVars = u_upper.size();

		// �������������
		vector<GRBVar> xVars(numVars);
		for (int i = 0; i < numVars; ++i) {
			xVars[i] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
		}

		// ����������Ĳ�ȷ���Ա��� u
		vector<GRBVar> uVars(numuVars);
		for (int i = 0; i < numuVars; ++i) {
			uVars[i] = model.addVar(u_lower[i], u_upper[i], 0.0, GRB_CONTINUOUS);
		}

		// ����������Ŀ�꺯����min b^T x
		GRBLinExpr obj = 0;
		for (int i = 0; i < numVars; ++i) {
			obj += b[i] * xVars[i];
		}
		model.setObjective(obj, GRB_MINIMIZE);

		// ���Լ�� Gx >= h - Ey - Mu
		for (int i = 0; i < numConstrs; ++i) {
			GRBLinExpr constr = 0;
			for (int j = 0; j < numVars; ++j) {
				constr += G[i][j] * xVars[j];
			}
			double rhs = h[i];
			for (int j = 0; j < y.size(); ++j) {
				rhs -= E[i][j] * y[j];
			}
			model.addConstr(constr >= rhs);
		}

		// ���ò�ȷ���Լ���Լ��
		for (int i = 0; i < numuVars; ++i) {
			model.addConstr(uVars[i] >= u_lower[i]);
			model.addConstr(uVars[i] <= u_upper[i]);
		}

		// �Ż�ģ��
		model.optimize();

		// ��ȡ��ż��
		pi.resize(numConstrs);
		for (int i = 0; i < numConstrs; ++i) {
			pi[i] = model.getConstr(i).get(GRB_DoubleAttr_Pi);
		}

		// ��ȡu�Ľ�
		for (int i = 0; i < numuVars; ++i) {
			u[i] = uVars[i].get(GRB_DoubleAttr_X);
		}

		return model.get(GRB_DoubleAttr_ObjVal);
	}
	catch (GRBException& e) {
		cerr << "Error code = " << e.getErrorCode() << endl;
		cerr << e.getMessage() << endl;
		return GRB_INFINITY;
	}
}


double solveSubProblem1(GRBEnv& env, const vector<double>& y, vector<double>& pi, vector<double>& u) {
	try {
		GRBModel model = GRBModel(env);

		int numxVars = G[0].size();
		int numConstrs = G.size();
		int numyVars = y.size();
		int numuVars = u.size();
		
		//����������(��תΪ��ż����)����pi
		vector<GRBVar> piVars(numConstrs);
		for (size_t i = 0; i < numConstrs; i++) {
			piVars[i] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
		}

		// ����������Ĳ�ȷ���Ա��� u
		vector<GRBVar> uVars(numuVars);
		for (int i = 0; i < numuVars; ++i) {
			uVars[i] = model.addVar(u_lower[i], u_upper[i], 0.0, GRB_CONTINUOUS);
		}
		
		//����������Ŀ�꺯����  sum{ (h[i] - E[i] * yVal - M[i] * uVal) * pi[i] } �����ڻ�
		GRBLinExpr obj = 0.0;
		for (size_t i = 0; i < numConstrs; i++){
			double rhs = h[i];
			for (size_t j = 0; j < numyVars; j++){
				rhs -= E[i][j] * y[j];
			}
			for (size_t k = 0; k < numuVars; k++) {
				rhs -= M[i][k] * u[k];
			}
			obj += piVars[i] * rhs;
		}
		model.setObjective(obj, GRB_MAXIMIZE);

		//	����������Լ��
		for (size_t i = 0; i < numxVars; i++){
			GRBLinExpr constr = 0.0;
			for (size_t j = 0; j < numConstrs; j++){
				constr += G[j][i] * piVars[j];
			}
			model.addConstr(constr <= b[i]);
		}


		model.optimize();

		// ��ȡ��ż�����Ľ�
		pi.resize(numConstrs);
		for (int i = 0; i < numConstrs; ++i) {
			pi[i] = piVars[i].get(GRB_DoubleAttr_X);
		}

		// ��ȡ u �Ľ�
		u.resize(numuVars);
		for (int i = 0; i < numuVars; ++i) {
			u[i] = uVars[i].get(GRB_DoubleAttr_X);
		}

		return model.get(GRB_DoubleAttr_ObjVal);
	}

	catch (GRBException& e) {
		cerr << "Error Code: " << e.getErrorCode() << endl;
		cerr << "Error Info: " << e.getMessage() << endl;
	}
}

int main() {
	try {
		GRBEnv env = GRBEnv();
		vector<double> y(c.size(), 0.0);
		double eta = 0.0;

		vector<vector<double>> cuts;

		double LB = -GRB_INFINITY;
		double UB = GRB_INFINITY;

		int maxIterations = 100;
		double tolerance = 1e4;
		double gap = 0.0;

		for (size_t i = 0; i < maxIterations; i++){
			double masterObj = solveMasterProblem(env, y, eta, cuts);

			vector<double> pi;
			vector<double>u(u_lower.size(), 0);
			double subObj = solveSubProblem(env, y, pi, u);

			UB = min(UB, masterObj);
			LB = max(LB, c[0] * y[0] + c[1] * y[1] + eta);
			gap = (UB - LB) / abs(UB);
			
			if (gap < tolerance) {
				break;
			}

			vector<double> cut(y.size() + 1);
			for (size_t i = 0; i < y.size(); i++){
				cut[i] = pi[i];
			}
			cut.back() = subObj;
			cuts.push_back(cut);
		}
		        // ������Ž�
        cout << "Optimal value: " << UB << endl;
        cout << "Optimal y: ";
        for (double val : y) {
            cout << val << " ";
        }
        cout << endl;
        cout << "Optimal eta: " << eta << endl;


	}

	catch (GRBException& e) {
		cerr << "Error Code : " << e.getErrorCode() << endl;
		cerr << "Error Info : " << e.getMessage() << endl;
	}
	return EXIT_SUCCESS;
}