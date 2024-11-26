
program test

    use libflit
    use librgm

    implicit none

    type(rgm3_elastic) :: p
    integer :: i
    integer :: nt, nv, nm
    real, allocatable, dimension(:) :: rmo, fs, noise, smooth, height, height2
    real, allocatable, dimension(:) :: slope, lwv, lwv2, nsmooth1, nsmooth2, nsmooth3
    integer, allocatable, dimension(:) :: nf, nl !, pm
    real, allocatable, dimension(:, :, :) :: ppick, rmask, pf, rp
    integer :: i1, i2, i3, l1, l
    integer :: pick1, pick2, pick3
    integer :: dist
    integer, allocatable, dimension(:) :: nmeq
    integer, allocatable, dimension(:, :) :: blkrange
    character(len=1024) :: dir_output
    real, allocatable, dimension(:) :: p1, p2, p3
    real :: sd
    integer, allocatable, dimension(:) :: seeds, mf

    call mpistart

    call getpar_string('outdir', dir_output, './dataset')
    call getpar_int('ntrain', nt, 2000)
    call getpar_int('nvalid', nv, 200)
    call getpar_int('seed', l, 111)
    nm = nt + nv

    if (l == -1) then
        seeds = zeros(200) - 1
    else
        seeds = irandom(200, range=[1, 10*nm], seed=l)
    end if

    call make_directory(tidy(dir_output)//'/data_train')
    call make_directory(tidy(dir_output)//'/data_valid')
    call make_directory(tidy(dir_output)//'/target_train')
    call make_directory(tidy(dir_output)//'/target_valid')

   	fs = random(nm, range=[180.0, 260.0], seed=seeds(1))
   	nl = nint(128/(1.0/fs/0.001))
    nf = irandom(nm, range=[3, 16], seed=seeds(3))

    noise = random(nm, range=[0.05, 0.35], seed=seeds(6))
    smooth = random(20*nm, dist='normal', mu=0.0, sigma=2.0, seed=seeds(7))
    smooth = pack(smooth, mask=(smooth >= 0.0))
    where (smooth <= 0.5)
    	smooth = 0.0
    end where
    smooth = pack(smooth, mask=(smooth <= 6.5))
    nsmooth1 = smooth(1:nm)
    nsmooth2 = smooth(nm + 1:2*nm)
    nsmooth3 = smooth(2*nm + 1:3*nm)

    smooth = random(nm, range=[10.0, 20.0], seed=seeds(10))
    height = random(nm, range=[2.0, 20.0], seed=seeds(11))
    height2 = random(nm, range=[15.0, 25.0], seed=seeds(12))
    lwv = random(nm, range=[0.0, 0.2], seed=seeds(13))
    lwv2 = random(nm, range=[0.3, 0.7]/3.0, seed=seeds(14))
    slope = random(nm, range=[-25.0, 25.0], seed=seeds(15))
    p%dip = [50.0, 130.0]
    p%strike = [0.0, 180.0]
    p%rake = [0.0, 180.0]

    nmeq = irandom(nm, range=[50, 1000], seed=seeds(16))
    nmeq(1:nint(nm*0.2)) = 0
    nmeq = random_permute(nmeq, seed=seeds(17))
    rmo = random(nm, range=[0.1, 0.6], seed=seeds(18))

    call alloc_array(blkrange, [0, nrank - 1, 1, 2])
    call cut(1, nm, nrank, blkrange)

    do i = blkrange(rankid, 1), blkrange(rankid, 2)

        p%seed = seeds(19) + i

        p%n1 = 128
        p%n2 = 128
        p%n3 = 128
        p%nf = nf(i)
        p%refl_slope = slope(i)
        p%f0 = fs(i)
        p%nl = nl(i)
        p%refl_amp = [0.05, 1.0]
        p%fwidth = 2.0
        p%yn_conv_noise = .false.
        if (mod(irand(range=[1, nm], seed=seeds(20) + i), 3) == 0) then
            p%refl_shape = 'gaussian'
            p%refl_mu2 = [0.0, p%n2 - 1.0]
            p%refl_mu3 = [0.0, p%n3 - 1.0]
            p%refl_sigma2 = [25.0, 60.0]
            p%refl_sigma3 = [25.0, 60.0]
            p%ng = irand(range=[1, 4], seed=seeds(21) + i)
            p%refl_height = [0.0, height2(i)]
            p%lwv = lwv2(i)
            p%secondary_refl_height_ratio = 0.05
        else
            p%refl_shape = 'random'
            p%refl_smooth = smooth(i)
            p%refl_height = [0.0, height(i)]
            p%lwv = lwv(i)
            p%secondary_refl_height_ratio = 0.1
        end if

        select case (mod(irand(range=[1, nm], seed=seeds(22) + i), 4))
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

        if (mod(irand(range=[1, nm], seed=seeds(23) + i), 3) == 0) then
            p%refl_amp_dist = 'uniform'
        else
            p%refl_amp_dist = 'normal'
        end if

        if (mod(irand(range=[1, nm], seed=seeds(24) + i), 3) == 0) then
            p%unconf = irand(range=[1, 2], seed=seeds(25) + i)
            p%unconf_z = [0.01, 0.1]
            p%unconf_amp = [0.05, 0.1]
        else
            p%unconf = 0
        end if

        if (nf(i) > 0.666*maxval(nf)) then
            p%yn_regular_fault = .true.
            p%nf = nf(i)
            if (mod(irand(range=[1, 10], seed=seeds(26) + i), 2) == 0) then
                p%dip = [rand(range=[100.0, 120.0], seed=seeds(27) + i), rand(range=[60.0, 80.0], seed=seeds(28) + i)]
                p%strike = [rand(range=[0.0, 90.0], seed=seeds(29) + i), rand(range=[90.0, 180.0], seed=seeds(29) + i)]
                p%rake = [rand(range=[0.0, 90.0], seed=seeds(30) + i), rand(range=[90.0, 180.0], seed=seeds(31) + i)]
                p%disp = [4.0, -4.0]
            else
                p%dip = [rand(range=[60.0, 80.0], seed=seeds(32) + i), rand(range=[100.0, 120.0], seed=seeds(33) + i)]
                p%strike = [rand(range=[90.0, 180.0], seed=seeds(34) + i), rand(range=[0.0, 90.0], seed=seeds(35) + i)]
                p%rake = [rand(range=[90.0, 180.0], seed=seeds(36) + i), rand(range=[0.0, 90.0], seed=seeds(37) + i)]
                p%disp = [-4.0, 4.0]
            end if
        else
            p%yn_regular_fault = .false.
            p%nf = nf(i)
            p%disp = [2.0, 10.0]
            p%dip = [55.0, 125.0]
            p%strike = [0.0, 180.0]
            p%rake = [0.0, 180.0]
        end if

        ! Clean image
        p%noise_level = noise(i)
        sd = rand(range=[0.5, 2.5], seed=seeds(38) + i)
        p%psf_sigma = [16.0, sd, sd]
        p%yn_fault = .true.
        p%image_smooth = [0.0, 0.0, 0.0]
        
       	if (mod(irand(range=[1, nm], seed=seeds(88) + i), 2) == 0) then
       		p%noise_type = 'uniform'
       	else
       		p%noise_type = 'normal'
       	end if
       	
       	if (mod(irand(range=[1, nm], seed=seeds(99) + i), 4) == 0) then
       		p%yn_conv_noise = .true.
       	else
       		p%yn_conv_noise = .false.
       	end if
       	
       	p%noise_smooth = [nsmooth1(i), nsmooth2(i), nsmooth3(i)]

        p%yn_facies = .false.
        p%vpvsratiomin = sqrt(2.5)
        p%vpvsratiomax = sqrt(3.5)
        p%perturb_max = 0.5

        call p%generate

        ! MEQ
        ppick = zeros(p%n1, p%n2, p%n3)

        if (nmeq(i) >= 1) then

            pf = p%fault

            ! Remove seismicity associated with some of the faults
            mf = irandom(nint(0.333*p%nf), range=[1, p%nf], seed=seeds(66 + i))
            do l = 1, size(mf)
                where (pf == mf(l))
                    pf = 0
                end where
            end do

            ! Randomly remove seismicity
            rmask = random_mask_smooth(p%n1, p%n2, p%n3, gs=[10.0, 10.0, 10.0], mask_out=rmo(i), seed=seeds(39) + i)
            pf = pf*rmask

            l1 = 0
            do l = 1, 10*maxval(nmeq)

                if (l1 < nmeq(i)) then

                    if (p%yn_regular_fault) then
                        dist = irand(range=[0, 2], seed=seeds(40) + l)
                    else
                        dist = irand(range=[0, 5], seed=seeds(41) + l)
                    end if

                    pick1 = irand(range=[dist + 1, p%n1 - dist], seed=seeds(42) + l)
                    pick2 = irand(range=[dist + 1, p%n2 - dist], seed=seeds(43) + l)
                    pick3 = irand(range=[dist + 1, p%n3 - dist], seed=seeds(44) + l)

                    if (any(pf(pick1 - dist:pick1 + dist, pick2 - dist:pick2 + dist, pick3 - dist:pick3 + dist) == 1)) then
                        dist = 4
                        do i3 = -2*dist - 1, 2*dist + 1
                            do i2 = -2*dist - 1, 2*dist + 1
                                do i1 = -2*dist - 1, 2*dist + 1
                                    if (pick1 + i1 >= 1 .and. pick1 + i1 <= p%n1 &
                                            .and. pick2 + i2 >= 1 .and. pick2 + i2 <= p%n2 &
                                            .and. pick3 + i3 >= 1 .and. pick3 + i3 <= p%n3) then
                                        ppick(pick1 + i1, pick2 + i2, pick3 + i3) = &
                                            max(ppick(pick1 + i1, pick2 + i2, pick3 + i3), exp(-0.3*(i1**2 + i2**2 + i3**2)))
                                    end if
                                end do
                            end do
                        end do
                        l1 = l1 + 1
                    end if

                end if

            end do

            ! Add seismicity noise
            dist = 1
            p1 = random(nint(0.1*nmeq(i)), range=[1, p%n1]*1.0, seed=seeds(45) + i)
            p2 = random(nint(0.1*nmeq(i)), range=[1, p%n2]*1.0, seed=seeds(46) + i)
            p3 = random(nint(0.1*nmeq(i)), range=[1, p%n3]*1.0, seed=seeds(47) + i)

            do l = 1, size(p1)
                pick1 = nint(p1(l)) + 1
                pick2 = nint(p2(l)) + 1
                pick3 = nint(p3(l)) + 1
                !$omp parallel do private(i1, i2, i3)
                do i3 = -2*dist - 1, 2*dist + 1
                    do i2 = -2*dist - 1, 2*dist + 1
                        do i1 = -2*dist - 1, 2*dist + 1
                            if (pick1 + i1 >= 1 .and. pick1 + i1 <= p%n1 &
                                    .and. pick2 + i2 >= 1 .and. pick2 + i2 <= p%n2 &
                                    .and. pick3 + i3 >= 1 .and. pick3 + i3 <= p%n3) then
                                ppick(pick1 + i1, pick2 + i2, pick3 + i3) = &
                                    max(ppick(pick1 + i1, pick2 + i2, pick3 + i3), exp(-0.3*(i1**2 + i2**2 + i3**2)))
                            end if
                        end do
                    end do
                end do
                !$omp end parallel do
            end do

        end if

        ! Make fault probability to one
        where (p%fault /= 0)
            p%fault = 1.0
        end where

        ! Normalize
        rp = random(p%n1, p%n2, p%n3, seed=seeds(48) + i)
        rp = gauss_filt(rp, [1, 1, 1]*rand(range=[3.0, 10.0], seed=seeds(49) + i))
        rp = rescale(rp, [0.3333, 1.0])
        p%image_pp = p%image_pp*rp
        p%image_ps = p%image_ps*rp
        p%image_sp = p%image_sp*rp
        p%image_ss = p%image_ss*rp

        sd = std(p%image_pp)
        p%image_pp = p%image_pp/sd
        p%image_ps = p%image_ps/sd
        p%image_sp = p%image_sp/sd
        p%image_ss = p%image_ss/sd

        if (i <= nt) then
            call output_array(p%image_pp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_pp.bin')
            call output_array(p%image_pp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img.bin')
            call output_array(p%image_ps, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_ps.bin')
            call output_array(p%image_sp, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_sp.bin')
            call output_array(p%image_ss, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_img_ss.bin')
            call output_array(ppick, tidy(dir_output)//'/data_train/'//num2str(i - 1)//'_meq.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fdip.bin')
            call output_array(p%fault_strike/180.0, tidy(dir_output)//'/target_train/'//num2str(i - 1)//'_fstrike.bin')

        else
            call output_array(p%image_pp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_pp.bin')
            call output_array(p%image_pp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img.bin')
            call output_array(p%image_ps, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_ps.bin')
            call output_array(p%image_sp, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_sp.bin')
            call output_array(p%image_ss, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_img_ss.bin')
            call output_array(ppick, tidy(dir_output)//'/data_valid/'//num2str(i - nt - 1)//'_meq.bin')

            call output_array(p%fault, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fsem.bin')
            call output_array(p%fault_dip/180.0, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fdip.bin')
            call output_array(p%fault_strike/180.0, tidy(dir_output)//'/target_valid/'//num2str(i - nt - 1)//'_fstrike.bin')
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
