for test_subject = 1:9
  for sub_idx = 1:9
    if sub_idx~=test_subject
      printf('%d, %d', sub_idx, test_subject)
      printf('\n')
    end
  end
end