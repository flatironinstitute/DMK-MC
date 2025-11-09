#include <hpdmk.h>

#include <cstdio>
#include <exception>
#include <stdexcept>

#include <tree.hpp>
#include <sctl.hpp>

namespace hpdmk {
    template <typename Real>
    inline hpdmk_tree create_tree(MPI_Comm comm, HPDMKParams params, int n_src, const Real *r_src, const Real *charge) {
        if (n_src < 0) {
            throw std::invalid_argument("number of sources must be non-negative");
        }
        if (n_src > 0 && (!r_src || !charge)) {
            throw std::invalid_argument("source and charge pointers must be non-null");
        }

        const sctl::Comm sctl_comm(comm);

        sctl::Vector<Real> r_src_vec(n_src * 3, const_cast<Real *>(r_src), false);
        sctl::Vector<Real> charge_vec(n_src, const_cast<Real *>(charge), false);

        auto *tree = new hpdmk::HPDMKPtTree<Real>(sctl_comm, params, r_src_vec, charge_vec);
        return static_cast<hpdmk_tree>(tree);
    }

    template <typename Real>
    inline void destroy_tree(hpdmk_tree tree) {
        if (!tree) {
            return;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        delete tree_ptr;
    }

    template <typename Real>
    inline void form_outgoing_pw(hpdmk_tree tree) {
        if (!tree) {
            return;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        tree_ptr->form_outgoing_pw();
    }

    template <typename Real>
    inline void form_incoming_pw(hpdmk_tree tree) {
        if (!tree) {
            return;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        tree_ptr->form_incoming_pw();
    }

    template <typename Real>
    inline Real eval_energy(hpdmk_tree tree) {
        if (!tree) {
            return 0;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        return tree_ptr->eval_energy();
    }

    template <typename Real>
    inline Real eval_energy_window(hpdmk_tree tree) {
        if (!tree) {
            return 0;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        return tree_ptr->eval_energy_window();
    }

    template <typename Real>
    inline Real eval_energy_diff(hpdmk_tree tree) {
        if (!tree) {
            return 0;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        return tree_ptr->eval_energy_diff();
    }

    template <typename Real>
    inline Real eval_energy_res(hpdmk_tree tree) {
        if (!tree) {
            return 0;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        return tree_ptr->eval_energy_res();
    }

    template <typename Real>
    inline Real eval_shift_energy(hpdmk_tree tree, long long i_particle, Real dx, Real dy, Real dz) {
        if (!tree) {
            return 0;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        return tree_ptr->eval_shift_energy(static_cast<sctl::Long>(i_particle), dx, dy, dz);
    }

    template <typename Real>
    inline void update_shift(hpdmk_tree tree, long long i_particle, Real dx, Real dy, Real dz) {
        if (!tree) {
            return;
        }
        auto *tree_ptr = static_cast<hpdmk::HPDMKPtTree<Real> *>(tree);
        tree_ptr->update_shift(static_cast<sctl::Long>(i_particle), dx, dy, dz);
    }
}

extern "C" {
    hpdmk_tree hpdmk_tree_create(MPI_Comm comm, HPDMKParams params, int n_src, const double *r_src, const double *charge) {
        try {
            return hpdmk::create_tree<double>(comm, params, n_src, r_src, charge);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_create failed: %s\n", ex.what());
        } catch (...) {
            std::fprintf(stderr, "hpdmk_tree_create failed due to an unknown exception\n");
        }
        return nullptr;
    }

    void hpdmk_tree_destroy(hpdmk_tree tree) {
        try {
            hpdmk::destroy_tree<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_destroy failed: %s\n", ex.what());
        }
    }

    void hpdmk_tree_form_outgoing_pw(hpdmk_tree tree) {
        try {
            hpdmk::form_outgoing_pw<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_form_outgoing_pw failed: %s\n", ex.what());
        }
    }

    void hpdmk_tree_form_incoming_pw(hpdmk_tree tree) {
        try {
            hpdmk::form_incoming_pw<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_form_incoming_pw failed: %s\n", ex.what());
        }
    }

    double hpdmk_eval_energy(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy failed: %s\n", ex.what());
        }
        return 0.0;
    }

    double hpdmk_eval_energy_window(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_window<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_window failed: %s\n", ex.what());
        }
        return 0.0;
    }

    double hpdmk_eval_energy_diff(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_diff<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_diff failed: %s\n", ex.what());
        }
        return 0.0;
    }

    double hpdmk_eval_energy_res(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_res<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_res failed: %s\n", ex.what());
        }
        return 0.0;
    }

    double hpdmk_eval_shift_energy(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz) {
        try {
            return hpdmk::eval_shift_energy<double>(tree, i_particle, dx, dy, dz);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_shift_energy failed: %s\n", ex.what());
        }
        return 0.0;
    }

    void hpdmk_update_shift(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz) {
        try {
            hpdmk::update_shift<double>(tree, i_particle, dx, dy, dz);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_update_shift failed: %s\n", ex.what());
        }
    }
}
