#include <hpdmk.h>

#include <cstdio>
#include <exception>
#include <stdexcept>

#include <tree.hpp>
#include <sctl.hpp>

namespace hpdmk {
    inline void ensure_mpi_initialized() {
        int initialized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Initialized failed");
        }
        if (!initialized) {
            int provided = MPI_THREAD_SINGLE;
            if (MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Init_thread failed");
            }
        }
    }

    template <typename Real>
    inline MPI_Comm normalize_comm(MPI_Comm comm) {
        if (comm == MPI_COMM_NULL) {
            return MPI_COMM_WORLD;
        }
        return comm;
    }

    inline hpdmk_tree create_tree(MPI_Comm comm, HPDMKParams params, int n_src, const Real *r_src, const Real *charge) {
        if (n_src < 0) {
            throw std::invalid_argument("number of sources must be non-negative");
        }
        if (n_src > 0 && (!r_src || !charge)) {
            throw std::invalid_argument("source and charge pointers must be non-null");
        }

        const sctl::Comm sctl_comm(normalize_comm(comm));

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
    int hpdmk_mpi_init(void) {
        try {
            hpdmk::ensure_mpi_initialized();
            return 1;
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_mpi_init failed: %s\n", ex.what());
        } catch (...) {
            std::fprintf(stderr, "hpdmk_mpi_init failed due to an unknown exception\n");
        }
        return 0;
    }

    int hpdmk_mpi_initialized(void) {
        int initialized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
            return 0;
        }
        return initialized;
    }

    MPI_Comm hpdmk_comm_world(void) {
        return MPI_COMM_WORLD;
    }

    hpdmk_tree hpdmk_tree_create(MPI_Comm comm, HPDMKParams params, int n_src, const double *r_src, const double *charge) {
        try {
            hpdmk::ensure_mpi_initialized();
            return hpdmk::create_tree<double>(comm, params, n_src, r_src, charge);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_create failed: %s\n", ex.what());
        } catch (...) {
            std::fprintf(stderr, "hpdmk_tree_create failed due to an unknown exception\n");
        }
        return nullptr;
    }

    hpdmk_tree hpdmk_tree_create_f(MPI_Comm comm, HPDMKParams params, int n_src, const float *r_src, const float *charge) {
        try {
            hpdmk::ensure_mpi_initialized();
            return hpdmk::create_tree<float>(comm, params, n_src, r_src, charge);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_create_f failed: %s\n", ex.what());
        } catch (...) {
            std::fprintf(stderr, "hpdmk_tree_create_f failed due to an unknown exception\n");
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

    void hpdmk_tree_destroy_f(hpdmk_tree tree) {
        try {
            hpdmk::destroy_tree<float>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_destroy_f failed: %s\n", ex.what());
        }
    }

    void hpdmk_tree_form_outgoing_pw(hpdmk_tree tree) {
        try {
            hpdmk::form_outgoing_pw<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_form_outgoing_pw failed: %s\n", ex.what());
        }
    }

    void hpdmk_tree_form_outgoing_pw_f(hpdmk_tree tree) {
        try {
            hpdmk::form_outgoing_pw<float>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_form_outgoing_pw_f failed: %s\n", ex.what());
        }
    }

    void hpdmk_tree_form_incoming_pw(hpdmk_tree tree) {
        try {
            hpdmk::form_incoming_pw<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_form_incoming_pw failed: %s\n", ex.what());
        }
    }

    void hpdmk_tree_form_incoming_pw_f(hpdmk_tree tree) {
        try {
            hpdmk::form_incoming_pw<float>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_tree_form_incoming_pw_f failed: %s\n", ex.what());
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

    float hpdmk_eval_energy_f(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy<float>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_f failed: %s\n", ex.what());
        }
        return 0.0f;
    }

    double hpdmk_eval_energy_window(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_window<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_window failed: %s\n", ex.what());
        }
        return 0.0;
    }

    float hpdmk_eval_energy_window_f(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_window<float>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_window_f failed: %s\n", ex.what());
        }
        return 0.0f;
    }

    double hpdmk_eval_energy_diff(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_diff<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_diff failed: %s\n", ex.what());
        }
        return 0.0;
    }

    float hpdmk_eval_energy_diff_f(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_diff<float>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_diff_f failed: %s\n", ex.what());
        }
        return 0.0f;
    }

    double hpdmk_eval_energy_res(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_res<double>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_res failed: %s\n", ex.what());
        }
        return 0.0;
    }

    float hpdmk_eval_energy_res_f(hpdmk_tree tree) {
        try {
            return hpdmk::eval_energy_res<float>(tree);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_energy_res_f failed: %s\n", ex.what());
        }
        return 0.0f;
    }

    double hpdmk_eval_shift_energy(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz) {
        try {
            return hpdmk::eval_shift_energy<double>(tree, i_particle, dx, dy, dz);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_shift_energy failed: %s\n", ex.what());
        }
        return 0.0;
    }

    float hpdmk_eval_shift_energy_f(hpdmk_tree tree, long long i_particle, float dx, float dy, float dz) {
        try {
            return hpdmk::eval_shift_energy<float>(tree, i_particle, dx, dy, dz);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_eval_shift_energy_f failed: %s\n", ex.what());
        }
        return 0.0f;
    }

    void hpdmk_update_shift(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz) {
        try {
            hpdmk::update_shift<double>(tree, i_particle, dx, dy, dz);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_update_shift failed: %s\n", ex.what());
        }
    }

    void hpdmk_update_shift_f(hpdmk_tree tree, long long i_particle, float dx, float dy, float dz) {
        try {
            hpdmk::update_shift<float>(tree, i_particle, dx, dy, dz);
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "hpdmk_update_shift_f failed: %s\n", ex.what());
        }
    }
}
