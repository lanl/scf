
program test

    use libflit
    use librgm

    implicit none

    type(rgm2_elastic) :: p
    integer :: i
    integer :: nt, nv, nm
    real, allocatable, dimension(:) :: rmo, fs, noise, smooth, height, height2
    real, allocatable, dimension(:) :: slope, lwv, lwv2, nsmooth1, nsmooth2
    integer, allocatable, dimension(:) :: nf, nl, pm
    real, allocatable, dimension(:, :) :: ppick, rmask, pf
    integer :: i1, i2, l1, l
    integer :: pick1, pick2
    integer :: dist
    integer, allocatable, dimension(:) :: nmeq
    integer, allocatable, dimension(:, :) :: blkrange
    character(len=1024) :: dir_output

    call mpistart

    call getpar_string('outdir', dir_output, './dataset')
    call getpar_int('ntrain', nt, 6000)
    call getpar_int('nvalid', nv, 600)
    nm = nt + nv

    call make_directory(tidy(dir_output)//'/data_train')
    call make_directory(tidy(dir_output)//'/data_valid')
    call make_directory(tidy(dir_output)//'/target_train')
    call make_directory(tidy(dir_output)//'/target_valid')

    fs = random(nm, range=[120.0, 220.0])
    nl = [nint(rescale(fs(1:nm/2), range=[30.0, 55.0])), nint(rescale(fs(nm/2 + 1:nm), range=[30.0, 55.0])*0.4)]*1.8
    pm = random_order(nm)
    fs = fs(pm)
    nl = nl(pm)

    nf = [irandom(nm/2, range=[1, 10]), irandom(nm/2, range=[10, 16])]
    pm = random_order(nm)
    nf = nf(pm)

    noise = random(nm, range=[0.0, 0.4])
    nsmooth1 = random(nm, range=[0.0, 1.5])
    nsmooth2 = random(nm, range=[0.0, 1.5])
    where (nsmooth1 < 0.5) nsmooth1 = 0.0
    where (nsmooth2 < 0.5) nsmooth2 = 0.0

    smooth = random(nm, range=[15.0, 25.0])
    height = random(nm, range=[2.0, 20.0])
    height2 = random(nm, range=[15.0, 25.0])
    lwv = random(nm, range=[0.0, 0.2])
    lwv2 = random(nm, range=[0.1, 0.5])
    slope = random(nm, range=[-25.0, 25.0])*2
    p%dip = [50.0, 130.0]

    nmeq = nint(rescale(nf*1.0, [50.0, 3000.0]))
    rmo = random(nm, range=[0.1, 0.6])

    call alloc_array(blkrange, [0, nrank - 1, 1, 2])
    call cut(1, nm, nrank, blkrange)

    do i = blkrange(rankid, 1), blkrange(rankid, 2)

        p%seed = i*i + 1000

        p%n1 = 256
        p%n2 = 256
        p%yn_conv_noise = .false.
        p%nf = nf(i)
        p%refl_slope = slope(i)
        p%f0 = fs(i)
        p%nl = nl(i)
        p%refl_amp = [0.1, 1.0]
        p%fwidth = 2.0
        if (mod(irand(range=[1, nm]), 3) == 0) then
            p%refl_shape = 'gaussian'
            p%refl_mu2 = [0.0, p%n2 - 1.0]
            p%refl_sigma2 = [40.0, 90.0]
            p%ng = irand(range=[2, 6])
            p%refl_height = [0.25*height2(i), height2(i)]
            p%lwv = lwv2(i)
            p%secondary_refl_amp = rand(range=[0.0, 0.1])
        else
            p%refl_shape = 'random'
            p%refl_smooth = 30
            p%refl_height = [0.0, height(i)]
            p%lwv = lwv(i)
            p%secondary_refl_amp = rand(range=[0.0, 0.2])
        end if

        select case (mod(irand(range=[1, nm]), 4))
            case (0)
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

        if (mod(irand(range=[1, nm]), 3) == 0) then
            p%refl_amp_dist = 'normal'
        else
            p%refl_amp_dist = 'uniform'
        end if

        if (mod(irand(range=[1, nm]), 3) == 0) then
            p%unconf = irand(range=[1, 2])
            p%unconf_z = [0.01, 0.5]
            p%unconf_amp = [0.05, 0.1]
        else
            p%unconf = 0
        end if

        if (nf(i) > 10) then
            p%yn_regular_fault = .true.
            p%nf = nf(i)
            if (mod(irand(range=[1, 10]), 2) == 0) then
                p%dip = [rand(range=[100.0, 120.0]), rand(range=[60.0, 80.0])]
                p%disp = [3.0, -3.0]
            else
                p%dip = [rand(range=[60.0, 80.0]), rand(range=[100.0, 120.0])]
                p%disp = [-3.0, 3.0]
            end if
        else
            p%yn_regular_fault = .false.
            p%nf = nf(i)
            p%disp = [5.0, 30.0]
            p%dip = [50.0, 130.0]
        end if

        ! Clean image
        p%yn_facies = .true.
        p%yn_fault = .true.

        if (mod(irand(range=[1, nm]), 2) == 0) then
            p%noise_type = 'normal'
        else
            p%noise_type = 'uniform'
        end if
        p%noise_smooth = [nsmooth1(i), nsmooth2(i)]

        p%psf_sigma = [12.0, 2.0]
        p%noise_level = noise(i)

        p%vpvsratiomin = sqrt(2.5)
        p%vpvsratiomax = sqrt(3.5)
        p%perturb_max = 0.5

        call p%generate

        where (p%fault /= 0)
            p%fault = 1.0
        end where

        ! MEQ
        rmask = random_mask_smooth(p%n1, p%n2, gs=[4.0, 4.0], mask_out=rmo(i))
        pf = p%fault*rmask
        ppick = zeros(p%n1, p%n2)
        l1 = 0
        do l = 1, 5*maxval(nmeq)

            if (l1 < nmeq(i)) then

                if (p%yn_regular_fault) then
                    dist = irand(range=[0, 2])
                else
                    dist = irand(range=[0, 5])
                end if

                pick1 = irand(range=[dist + 1, p%n1 - dist])
                pick2 = irand(range=[dist + 1, p%n2 - dist])

                if (any(pf(pick1 - dist:pick1 + dist, pick2 - dist:pick2 + dist) == 1)) then
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

            end if

        end do

        if (i <= nt) then
            call output_array(p%image_pp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_pp.bin')
            call output_array(p%image_pp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img.bin')
            call output_array(p%image_ps, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_ps.bin')
            call output_array(p%image_sp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_sp.bin')
            call output_array(p%image_ss, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_ss.bin')
            call output_array(ppick, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_meq.bin')

            call output_array(p%fault*rmask, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0*rmask, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_fdip.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fdip.bin')

        else
            call output_array(p%image_pp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_pp.bin')
            call output_array(p%image_pp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img.bin')
            call output_array(p%image_ps, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_ps.bin')
            call output_array(p%image_sp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_sp.bin')
            call output_array(p%image_ss, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_ss.bin')
            call output_array(ppick, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_meq.bin')
            call output_array(p%fault*rmask, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0*rmask, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_fdip.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fdip.bin')
        end if

        if (i <= nt) then
            print *, date_time_compact(), ' train', i - 1, l1
        else
            print *, date_time_compact(), ' valid', i - nt - 1, l1
        end if

    end do

    call mpibarrier
    call mpiend

end program test
