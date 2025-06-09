!
! (c) 2025. Triad National Security, LLC. All rights reserved.
!
! This program was produced under U.S. Government contract 89233218CNA000001
! for Los Alamos National Laboratory (LANL), which is operated by
! Triad National Security, LLC for the U.S. Department of Energy/National
! Nuclear Security Administration. All rights in the program are reserved
! by Triad National Security, LLC, and the U.S. Department of Energy/
! National Nuclear Security Administration.
! The Government is granted for itself and others acting on its behalf a nonexclusive,
! paid-up, irrevocable worldwide license in this material to reproduce, prepare,
! derivative works, distribute copies to the public, perform publicly
! and display publicly, and to permit others to do so.
!
! Author:
!   Kai Gao, kaigao@lanl.gov
!
! Functionality:
!   The script is to generate training dataset for training the 2D version of
!   F-Net, SCF-Net, and other variants of the models
!
! Note:
!   The script is a slight modification to the previous version by adding deterministic
!   random seeds to ensure reproducibility.
!
program test

    use libflit
    use librgm

    implicit none

    type(rgm2_elastic) :: p
    integer :: i
    integer :: nt, nv, nm
    real, allocatable, dimension(:) :: rmo, fs, noise, smooth, height, height2
    real, allocatable, dimension(:) :: slope, lwv, lwv2, nsmooth1, nsmooth2
    integer, allocatable, dimension(:) :: nf, nl
    real, allocatable, dimension(:, :) :: ppick, rmask, pf
    integer :: i1, i2, l1, l
    integer :: pick1, pick2
    integer :: dist
    integer, allocatable, dimension(:) :: nmeq
    integer, allocatable, dimension(:, :) :: blkrange
    real, allocatable, dimension(:) :: px, pz
    character(len=1024) :: dir_output
    real, allocatable, dimension(:, :) :: rp
    real :: sd
    integer, allocatable, dimension(:) :: seeds, mf
    real, allocatable, dimension(:) :: noise_scalar

    call mpistart

    call getpar_string('outdir', dir_output, './dataset')
    call getpar_int('ntrain', nt, 6000)
    call getpar_int('nvalid', nv, 600)
    call getpar_nfloat('noise_scalar', noise_scalar, [0.0, 1.0])
    nm = nt + nv

    call getpar_int('seed', l, 111)

    if (l == -1) then
        seeds = zeros(200) - 1
    else
        seeds = irandom(200, range=[1, 10*nm], seed=l)
    end if

    call make_directory(tidy(dir_output)//'/data_train')
    call make_directory(tidy(dir_output)//'/data_valid')
    call make_directory(tidy(dir_output)//'/target_train')
    call make_directory(tidy(dir_output)//'/target_valid')

    ! Frequency of wavelet for convolving with reflectivity series
    fs = random(nm, range=[140.0, 220.0], seed=seeds(1))

    ! Number of layers
    nl = rescale(fs, [40.0, 60.0])

    ! Number of faults
    nf = irandom(nm, range=[1, 16], seed=seeds(2))

    ! Noise level
    noise = random_pdf(nm, pdf=[0.1, 0.5, 1.0, 0.5], range=[0.0, 0.5], seed=seeds(3))*noise_scalar(2) + noise_scalar(1)
    if (rankid == 0) then
        call plot_histogram(noise)
    end if

    ! Noise smoothing
    nsmooth1 = random(nm, range=[0.0, 1.5], seed=seeds(4))
    nsmooth2 = random(nm, range=[0.0, 1.5], seed=seeds(5))
    where (nsmooth1 < 0.5)
        nsmooth1 = 0.0
    end where
    where (nsmooth2 < 0.5)
        nsmooth2 = 0.0
    end where

    ! Reflector properties
    smooth = random(nm, range=[10.0, 20.0], seed=seeds(6))
    height = random(nm, range=[2.0, 20.0], seed=seeds(7))
    height2 = random(nm, range=[15.0, 25.0], seed=seeds(8))/2.0
    lwv = random(nm, range=[0.0, 0.2], seed=seeds(9))
    lwv2 = random(nm, range=[0.1, 0.5], seed=seeds(10))/2.0

    ! Slope and dip
    slope = random(nm, range=[-25.0, 25.0], seed=seeds(11))*1.5
    p%dip = [50.0, 130.0]

    ! Number of seismicity
    nmeq = nint(rescale(nf*1.0, [25.0, 750.0]))
    rmo = random(nm, range=[0.1, 0.6], seed=seeds(12))

    call alloc_array(blkrange, [0, nrank - 1, 1, 2])
    call cut(1, nm, nrank, blkrange)

    do i = blkrange(rankid, 1), blkrange(rankid, 2)

        ! Deterministic random seeds for reproducibility
        p%seed = seeds(99) + i

        ! Dimension
        p%n1 = 256
        p%n2 = 256

        ! How to add noise
        if (mod(irand(range=[1, nm], seed=seeds(13) + i), 2) == 0) then
            p%yn_conv_noise = .true.
        else
            p%yn_conv_noise = .false.
        end if
        p%nf = nf(i)
        p%refl_slope = slope(i)
        p%f0 = fs(i)
        p%nl = nl(i)
        p%fwidth = 2.0

        ! Reflector shape
        if (mod(irand(range=[1, nm], seed=seeds(14) + i), 3) == 0) then
            p%refl_shape = 'gaussian'
            p%refl_mu2 = [0.0, p%n2 - 1.0]
            p%refl_sigma2 = [40.0, 90.0]/2.0
            p%ng = irand(range=[2, 6], seed=seeds(15) + i)
            p%refl_height = [0.25*height2(i), height2(i)]
            p%lwv = lwv2(i)
            p%secondary_refl_height_ratio = rand(range=[0.0, 0.1], seed=seeds(16) + i)
        else
            p%refl_shape = 'random'
            p%refl_smooth = 25
            p%refl_height = [0.0, height(i)]
            p%lwv = lwv(i)
            p%secondary_refl_height_ratio = rand(range=[0.0, 0.2], seed=seeds(17) + i)
        end if

        ! Wavelet shape
        select case (mod(irand(range=[1, nm], seed=seeds(18) + i), 5))
            case (0, 4)
                p%wave = 'ricker'
            case (1)
                p%wave = 'ricker_deriv'
            case (2)
                p%wave = 'gaussian_deriv'
            case (3)
                p%wave = 'delta'
                p%wave_filt_freqs = [0.0, 0.5*fs(i), 1.5*fs(i), 2.5*fs(i)]
                p%wave_filt_amps = [0.0, 1.0, 1.0, 0.0]
        end select

        ! Geological unconformity
        if (mod(irand(range=[1, nm], seed=seeds(19) + i), 3) == 0) then
            p%unconf = irand(range=[1, 2], seed=seeds(20) + i)
            p%unconf_z = [0.01, 0.5]
            p%unconf_amp = [0.05, 0.1]
        else
            p%unconf = 0
        end if

        ! Regular faults or randomly distributed faults
        if (nf(i) >= 12) then
            p%yn_regular_fault = .true.
            p%nf = nf(i)
            if (mod(irand(range=[1, 10], seed=seeds(21) + i), 2) == 0) then
                p%dip = [rand(range=[100.0, 120.0], seed=seeds(22) + i), rand(range=[60.0, 80.0], seed=seeds(23) + i)]
                p%disp = [4.0, -4.0]
            else
                p%dip = [rand(range=[60.0, 80.0], seed=seeds(24) + i), rand(range=[100.0, 120.0], seed=seeds(25) + i)]
                p%disp = [-4.0, 4.0]
            end if
        else
            p%yn_regular_fault = .false.
            p%nf = nf(i)
            p%disp = [5.0, 30.0]
            p%dip = [50.0, 130.0]
        end if

        ! Noise
        if (mod(irand(range=[1, nm], seed=seeds(26) + i), 2) == 0) then
            p%noise_type = 'normal'
        else
            p%noise_type = 'uniform'
        end if
        p%noise_smooth = [nsmooth1(i), nsmooth2(i)]

        p%psf_sigma = [16.0, 1.5]
        p%noise_level = noise(i)
        p%vpvsratiomin = sqrt(2.5)
        p%vpvsratiomax = sqrt(3.5)
        p%perturb_max = 0.5

        call p%generate

        ! Normalize generated images
        rp = random(p%n1, p%n2, seed=seeds(27) + i)
        rp = gauss_filt(rp, [1, 1]*rand(range=[1.0, 5.0], seed=seeds(28) + i))
        rp = rescale(rp, [0.25, 1.0])
        p%image_pp = p%image_pp*rp
        p%image_ps = p%image_ps*rp
        p%image_sp = p%image_sp*rp
        p%image_ss = p%image_ss*rp

        sd = std(p%image_pp)
        p%image_pp = p%image_pp/sd
        p%image_ps = p%image_ps/sd
        p%image_sp = p%image_sp/sd
        p%image_ss = p%image_ss/sd

        ! Generate seismicity location image
        ppick = zeros(p%n1, p%n2)

        if (nmeq(i) >= 1) then

            pf = p%fault

            ! Remove seismicity associated with some of the faults
            mf = irandom(nint(0.25*p%nf), range=[1, p%nf], seed=seeds(66) + i)
            do l = 1, size(mf)
                where (pf == mf(l))
                    pf = 0
                end where
            end do

            ! Randomly remove seismicity
            rmask = random_mask_smooth(p%n1, p%n2, gs=[10.0, 10.0], mask_out=rmo(i), seed=seeds(29) + i)
            pf = pf*rmask

            l1 = 0
            l = 0
            do while(l1 < nmeq(i) .and. l < 50*nmeq(i))

                dist = irand(range=[0, 2], seed=seeds(30) + i + l)

                pick1 = irand(range=[dist + 1, p%n1 - dist], seed=seeds(32) + i + l)
                pick2 = irand(range=[dist + 1, p%n2 - dist], seed=seeds(33) + i + l)

                if (any(pf(pick1 - dist:pick1 + dist, pick2 - dist:pick2 + dist) > 0)) then
                    dist = 3
                    do i2 = -2*dist - 1, 2*dist + 1
                        do i1 = -2*dist - 1, 2*dist + 1
                            if (pick1 + i1 >= 1 .and. pick1 + i1 <= p%n1 &
                                    .and. pick2 + i2 >= 1 .and. pick2 + i2 <= p%n2) then
                                ppick(pick1 + i1, pick2 + i2) = &
                                    max(ppick(pick1 + i1, pick2 + i2), exp(-0.3*(i1**2 + i2**2)))
                            end if
                        end do
                    end do
                    l1 = l1 + 1
                end if

                l = l + 1

            end do

            ! Add random seismicity noise that are not associated with faults
            dist = 1
            px = random(nint(0.05*nmeq(i)), range=[1, p%n1]*1.0, seed=seeds(34) + i)
            pz = random(nint(0.05*nmeq(i)), range=[1, p%n2]*1.0, seed=seeds(35) + i)

            do l = 1, size(px)
                pick1 = nint(px(l)) + 1
                pick2 = nint(pz(l)) + 1
                do i2 = -2*dist - 1, 2*dist + 1
                    do i1 = -2*dist - 1, 2*dist + 1
                        if (pick1 + i1 >= 1 .and. pick1 + i1 <= p%n1 &
                                .and. pick2 + i2 >= 1 .and. pick2 + i2 <= p%n2) then
                            ppick(pick1 + i1, pick2 + i2) = &
                                max(ppick(pick1 + i1, pick2 + i2), exp(-0.3*(i1**2 + i2**2)))
                        end if
                    end do
                end do
            end do

        end if

        where (p%fault /= 0)
            p%fault = 1.0
        end where

        ! Save generated data and labels
        if (i <= nt) then
            call output_array(p%image_pp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_pp.bin')
            call output_array(p%image_pp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img.bin')
            call output_array(p%image_ps, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_ps.bin')
            call output_array(p%image_sp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_sp.bin')
            call output_array(p%image_ss, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_ss.bin')
            call output_array(ppick, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_meq.bin')
            call output_array(p%fault, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fdip.bin')

        else
            call output_array(p%image_pp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_pp.bin')
            call output_array(p%image_pp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img.bin')
            call output_array(p%image_ps, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_ps.bin')
            call output_array(p%image_sp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_sp.bin')
            call output_array(p%image_ss, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_ss.bin')
            call output_array(ppick, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_meq.bin')
            call output_array(p%fault, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fdip.bin')
        end if

        if (i <= nt) then
            print *, date_time_compact(), ' train', i - 1, nmeq(i)
        else
            print *, date_time_compact(), ' valid', i - nt - 1, nmeq(i)
        end if

    end do

    call mpibarrier
    call mpiend

end program test
