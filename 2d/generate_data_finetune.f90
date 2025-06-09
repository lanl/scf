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
!   The script is to generate a small amount of fine tuning data for Opunake image.
!   The parameters are to mimic the geological features observed
!   in the Opunake image, and the dimension of the synthetic data
!   are consistent with it as well.
!
! Note:
!   The script is a slight modification to the previous version by adding deterministic
!   random seeds to ensure reproducibility.
!
program test

    use libflit
    use librgm

    implicit none

    type(rgm2_curved) :: p
    integer :: i
    integer :: nt, nv, nm
    real, allocatable, dimension(:) :: rmo, fs, noise
    real, allocatable, dimension(:) :: slope, nsmooth1, nsmooth2
    integer, allocatable, dimension(:) :: nf, nl, pm
    real, allocatable, dimension(:, :) :: ppick, rmask, pf
    integer :: i1, i2, l1, l
    integer :: pick1, pick2
    integer :: dist
    integer, allocatable, dimension(:) :: nmeq
    integer, allocatable, dimension(:, :) :: blkrange
    real, allocatable, dimension(:) :: px, pz
    character(len=1024) :: dir_output
    real, allocatable, dimension(:, :) :: rp
    integer, allocatable, dimension(:) :: seeds, mf

    call mpistart

    call getpar_string('outdir', dir_output, './dataset_finetune')
    call getpar_int('ntrain', nt, 300)
    call getpar_int('nvalid', nv, 30)
    nm = nt + nv

    call getpar_int('seed', l, 123)

    ! Seeds used for random number generation
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
    fs = random(nm, range=[130.0, 170.0], seed=seeds(1))

    ! Number of layers
    nl = nint(rescale(fs, range=[50.0, 70.0]))
    pm = random_order(nm, seed=seeds(2))
    fs = fs(pm)
    nl = nl(pm)

    ! Number of faults
    nf = irandom(nm, range=[15, 25], seed=seeds(3))
    pm = random_order(nm, seed=seeds(4))
    nf = nf(pm)

    ! Noise level
    noise = random(nm, range=[0.0, 0.1], seed=seeds(5))
    nsmooth1 = random(nm, range=[0.0, 1.5], seed=seeds(6))
    nsmooth2 = random(nm, range=[0.0, 1.5], seed=seeds(7))
    where (nsmooth1 < 0.5)
        nsmooth1 = 0.0
    end where
    where (nsmooth2 < 0.5)
        nsmooth2 = 0.0
    end where

    ! Slope and dip
    slope = random(nm, range=[-25.0, 25.0], seed=seeds(8))
    p%dip = [50.0, 130.0]

    ! Number of seismicity
    nmeq = nint(rescale(nf*1.0, [400.0, 1000.0]))
    rmo = random(nm, range=[0.1, 0.6], seed=seeds(9))

    call alloc_array(blkrange, [0, nrank - 1, 1, 2])
    call cut(1, nm, nrank, blkrange)

    do i = blkrange(rankid, 1), blkrange(rankid, 2)

        ! Set deterministic seed for reproducibility
        p%seed = seeds(10) + i

        p%n1 = 256
        p%n2 = 1024
        p%yn_conv_noise = .false.
        p%nf = nf(i)
        p%f0 = fs(i)
        p%nl = nl(i)
        p%fwidth = 2.0
        p%refl_shape = 'gaussian'
        p%refl_mu2 = [0.1*p%n2, 0.9*p%n2]
        p%refl_sigma2 = [200.0, 300.0]*0.75
        p%ng = 1
        p%refl_slope = slope(i)
        p%refl_height = [0, 75]

        ! Set smaller top slope and lateral variation
        p%refl_slope_top = slope(i)/5.0
        p%refl_smooth_top = 20
        p%refl_height_top = [0.0, 5.0]

        p%lwv = 0.25
        p%wave = 'ricker'

        if (mod(irand(range=[1, nm], seed=seeds(11) + i), 3) == 0) then
            p%unconf = irand(range=[1, 2], seed=seeds(12) + i)
            p%unconf_z = [0.01, 0.15]
            p%unconf_amp = [0.05, 0.1]
        else
            p%unconf = 0
        end if

        if (nf(i) > minval(nf) + 0.6666*rov(nf*1.0)) then
            p%yn_regular_fault = .true.
            p%nf = nf(i)
            if (mod(irand(range=[1, 1000], seed=seeds(13) + i), 2) == 0) then
                p%dip = [rand(range=[100.0, 120.0], seed=seeds(14) + i), rand(range=[60.0, 80.0], seed=seeds(15) + i)]
                p%disp = [4.0, -4.0]
            else
                p%dip = [rand(range=[60.0, 80.0], seed=seeds(14) + i), rand(range=[100.0, 120.0], seed=seeds(15) + i)]
                p%disp = [-4.0, 4.0]
            end if
        else
            p%yn_regular_fault = .false.
            p%nf = nf(i)
            p%disp = [4.0, 10.0]
            p%dip = [50.0, 130.0]
        end if

        if (mod(irand(range=[1, 1000], seed=seeds(13) + 2*i), 2) == 0) then
            p%delta_dip = [0.0, 5.0]
        else
            p%delta_dip = [0.0, 35.0]
        end if

        p%noise_smooth = [nsmooth1(i), nsmooth2(i)]
        p%psf_sigma = [15.0, 5.0]
        p%noise_level = noise(i)

        call p%generate

        rp = random(p%n1, p%n2, seed=seeds(16) + i, dist='gaussian')
        rp = gauss_filt(rp, [1, 1]*rand(range=[4.0, 14.0], seed=seeds(17) + i))
        rp = rescale(rp, [0.1, 1.0])
        p%image = p%image*rp
        p%image = p%image - mean(p%image)
        p%image = p%image/std(p%image)

        ! Generate source image
        ppick = zeros(p%n1, p%n2)

        if (nmeq(i) >= 1) then

            pf = p%fault

            ! Remove seismicity associated with some of the faults
            mf = irandom(nint(0.333*p%nf), range=[1, p%nf], seed=seeds(66) + i)
            do l = 1, size(mf)
                where (pf == mf(l))
                    pf = 0
                end where
            end do

            ! Randomly remove seismicity for some regions in space
            rmask = random_mask_smooth(p%n1, p%n2, gs=[4.0, 5.0], mask_out=rmo(i), seed=seeds(29) + i)
            pf = pf*rmask

            l1 = 0
            l = 0
            do while(l1 < nmeq(i) .and. l < 50*nmeq(i))

                if (p%yn_regular_fault) then
                    dist = irand(range=[0, 3], seed=seeds(30) + i + l)
                else
                    dist = irand(range=[0, 6], seed=seeds(31) + i + l)
                end if

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
            px = random(nint(0.1*nmeq(i)), range=[1, p%n1]*1.0, seed=seeds(34) + i)
            pz = random(nint(0.1*nmeq(i)), range=[1, p%n2]*1.0, seed=seeds(35) + i)

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

        ! Save data to training/validation directories
        if (i <= nt) then
            call output_array(p%image, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img.bin')
            call output_array(ppick, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_meq.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fdip.bin')

        else
            call output_array(p%image, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img.bin')
            call output_array(ppick, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_meq.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fdip.bin')
        end if

        ! Show progress
        if (i <= nt) then
            print *, date_time_compact(), ' train', i - 1, p%yn_regular_fault
        else
            print *, date_time_compact(), ' valid', i - nt - 1, p%yn_regular_fault
        end if

    end do

    call mpibarrier
    call mpiend

end program test
